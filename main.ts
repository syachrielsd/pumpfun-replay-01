// Cleanse-Recycle Kelly Compounder — Pump.fun Replay Sprint paper bot.
//
// Protocol: agent-fun/replay/v2. The harness sends a hello, then one batch per
// slot window, then done; the bot answers every batch with a reply carrying its
// orders, echoing the batch seq. All decisions derive only from the hello,
// batch and slot data — no clock, no randomness, no network — so two runs
// produce byte-identical action logs.
//
// Strategy (two-sleeve, in the convergent Kelly-compounder family):
//   Sleeve A "cleanse": enter only after an early-seller flush — the price
//     dipped >= 10% off its recent-window high and reclaimed >= 6% off the
//     low, after a >= 30% run-up, with buyers dominating the entry window
//     (>= 73% of recent SOL volume is buys), the price back above its
//     window mean, and fresh buy flow in the last trades.
//     Pullback-breakout, graduation-aware (no entries near completion).
//   Sleeve B "momentum-recycle": one second-wave re-entry per mint, only after
//     a profitable exit on that same mint, with a tighter stop and half size.
//   Exits: take-profit (+60%: let runners run), trailing stop (arms at +15%,
//     gives back 8pp), hard stop (-8% A / -6% B), bleed-out exit, max hold.
//   Sizing: Kelly-flavoured fraction of bankroll, capped per trade at
//     min(0.40 SOL, 100 bps of real SOL reserve) and by the 5-open-position
//     cap. Drawdown governor: after 2 consecutive losing coins, halve size
//     until a winning coin resets it.
//   Guardrails: every buy >= 0.012 SOL (round-trip bar is 0.01 SOL), at most
//     16 orders per batch, no shorts, no leverage, max slippage 1000 bps.

import {
  LAMPORTS_PER_SOL,
  runBot,
  type Batch,
  type BotState,
  type BuyInput,
  type Fill,
  type OrderInput,
  type SellInput,
} from './vendor/makefun-bot.ts';

// ---------------------------------------------------------------- parameters
const SCALE = 1_000_000_000n; // price scale: lamports per token * 1e9
const MIN_BUY_LAMPORTS = 12_000_000n; // 0.012 SOL, above the 0.01 round-trip bar
const MAX_STAKE_LAMPORTS = 400_000_000n; // 0.40 SOL hard cap per trade
const BASE_STAKE_LAMPORTS = 80_000_000n; // 0.08 SOL base Kelly stake
const MAX_SLIPPAGE_BPS = 1000;
const WINDOW_TRADES = 24; // buy-share / dip window length, in trades
const MIN_TRADES_TO_ENTER = 12;
const MIN_BUY_SHARE_BPS = 7300; // sleeve A: >= 73% buys in the window
const RECYCLE_BUY_SHARE_BPS = 7300; // sleeve B: >= 73% buys
const DIP_BPS = 1000; // >= 10% off the window high — a real flush, not noise
const RECLAIM_BPS = 600; // >= 6% up off the window low
const RUNUP_BPS = 3000; // window high >= 30% above the window's first price
const RECYCLE_DIP_BPS = 500;
const TAKE_PROFIT_BPS = 6000; // +60%: let runners run; the trailing stop banks them
const TRAIL_ARM_BPS = 1500; // +15% arms the trailing stop
const TRAIL_GIVEBACK_BPS = 800; // give back 8pp from the peak return
const HARD_STOP_A_BPS = 800; // -8% sleeve A: cut losers faster
const HARD_STOP_B_BPS = 600; // -6% sleeve B (tighter)
const BLEED_BPS = 300; // -3% after BLEED_SLOTS slots -> exit
const BLEED_SLOTS = 1000;
const MAX_HOLD_SLOTS = 2400;
const RECYCLE_WINDOW_SLOTS = 3000; // re-entry window after a profitable exit
const MIN_REAL_SOL_LAMPORTS = 500_000_000n; // 0.5 SOL real reserves to enter
const NEAR_COMPLETE_REAL_SOL = 70_000_000_000n; // 70 SOL: skip entries near graduation

// ---------------------------------------------------------------- per-mint state
interface MintTrack {
  prices: bigint[]; // recent window prices, scaled
  buySol: bigint[]; // per-trade buy SOL (window)
  sellSol: bigint[]; // per-trade sell SOL (window)
  trades: number;
  complete: boolean;
  sleeveAEntered: boolean;
  sleeveBEntered: boolean;
  lastProfitExitSlot: number;
  lastProfitExitPrice: bigint;
}

interface OpenPos {
  mint: string;
  entryPrice: bigint; // scaled lamports/token incl. buy fees
  entrySlot: number;
  sleeve: 'A' | 'B';
  peakRetBps: bigint;
  costLamports: bigint;
  tokens: bigint;
}

const tracks = new Map<string, MintTrack>();
// client_id -> pending buy meta (filled later from batch fills)
const pendingBuys = new Map<string, { mint: string; sleeve: 'A' | 'B'; slot: number }>();
// mints with a sell already sent and awaiting its fill: never double-sell,
// and never drop the position from tracking until the fill confirms it.
const pendingSells = new Set<string>();
const openPos = new Map<string, OpenPos>();
let losingStreak = 0; // consecutive losing coins, for the drawdown governor

function trackFor(mint: string): MintTrack {
  let t = tracks.get(mint);
  if (t === undefined) {
    t = {
      prices: [],
      buySol: [],
      sellSol: [],
      trades: 0,
      complete: false,
      sleeveAEntered: false,
      sleeveBEntered: false,
      lastProfitExitSlot: -1,
      lastProfitExitPrice: 0n,
    };
    tracks.set(mint, t);
  }
  return t;
}

function priceOf(state: BotState, mint: string): bigint {
  const c = state.curves.get(mint);
  if (c === undefined || c.virtual_token_reserves <= 0n) return 0n;
  return (c.virtual_quote_reserves * SCALE) / c.virtual_token_reserves;
}

function windowBuyShareBps(t: MintTrack): bigint {
  let buy = 0n;
  let total = 0n;
  for (let i = 0; i < t.buySol.length; i++) {
    buy += t.buySol[i];
    total += t.buySol[i] + t.sellSol[i];
  }
  if (total <= 0n) return 0n;
  return (buy * 10000n) / total;
}

function windowDipReclaim(t: MintTrack): { dipBps: bigint; reclaimBps: bigint; runupBps: bigint } {
  if (t.prices.length === 0) return { dipBps: 0n, reclaimBps: 0n, runupBps: 0n };
  let hi = 0n;
  let lo = 0n;
  for (const p of t.prices) {
    if (hi === 0n || p > hi) hi = p;
    if (lo === 0n || p < lo) lo = p;
  }
  const cur = t.prices[t.prices.length - 1];
  const first = t.prices[0];
  if (hi <= 0n || lo <= 0n || cur <= 0n || first <= 0n) {
    return { dipBps: 0n, reclaimBps: 0n, runupBps: 0n };
  }
  return {
    dipBps: ((hi - cur) * 10000n) / hi,
    reclaimBps: ((cur - lo) * 10000n) / lo,
    runupBps: ((hi - first) * 10000n) / first,
  };
}

// Fresh buy flow: how many of the last 3 window trades were buys.
function recentBuys(t: MintTrack, n: number): number {
  let c = 0;
  const start = Math.max(0, t.buySol.length - n);
  for (let i = start; i < t.buySol.length; i++) if (t.buySol[i] > 0n) c += 1;
  return c;
}

// Mean of the window prices: an entry only counts when the coin trades
// above its own recent average, so the reclaim is a real turn, not a bounce
// that still sits below the average.
function windowMeanPrice(t: MintTrack): bigint {
  if (t.prices.length === 0) return 0n;
  let sum = 0n;
  for (const p of t.prices) sum += p;
  return sum / BigInt(t.prices.length);
}

// Fold this batch's fills into our books: learn entry prices from buy fills
// and win/loss outcomes from sell fills (drives the governor + recycling).
function applyFills(batch: Batch, slot: number): void {
  for (const f of batch.fills as Fill[]) {
    if (f.status !== 'filled') {
      pendingBuys.delete(f.client_id);
      if (f.side === 'sell') pendingSells.delete(f.mint); // rejected sell: retry next batch
      continue;
    }
    if (f.side === 'buy') {
      const meta = pendingBuys.get(f.client_id);
      pendingBuys.delete(f.client_id);
      const tokens = BigInt(f.token_amount);
      const spent = BigInt(f.quote_lamports);
      if (tokens > 0n && spent > 0n) {
        openPos.set(f.mint, {
          mint: f.mint,
          entryPrice: (spent * SCALE) / tokens,
          entrySlot: meta !== undefined ? meta.slot : slot,
          sleeve: meta !== undefined ? meta.sleeve : 'A',
          peakRetBps: 0n,
          costLamports: spent,
          tokens,
        });
      }
    } else {
      const pos = openPos.get(f.mint);
      pendingSells.delete(f.mint);
      openPos.delete(f.mint);
      const proceeds = BigInt(f.quote_lamports);
      if (pos !== undefined) {
        if (proceeds > pos.costLamports) {
          losingStreak = 0;
          const t = trackFor(f.mint);
          t.lastProfitExitSlot = slot;
          t.lastProfitExitPrice = pos.entryPrice;
        } else {
          losingStreak += 1;
        }
      }
    }
  }
}

function stakeFor(state: BotState, buyShareBps: bigint, sleeve: 'A' | 'B'): bigint {
  // Kelly-flavoured: edge grows with buy share above 50%; base 0.06 SOL.
  const edge = buyShareBps > 5000n ? buyShareBps - 5000n : 0n; // 0..5000
  let stake = BASE_STAKE_LAMPORTS + (BASE_STAKE_LAMPORTS * edge) / 2500n;
  if (sleeve === 'B') stake /= 2n;
  if (losingStreak >= 2) stake /= 2n; // drawdown governor halves size
  if (stake > MAX_STAKE_LAMPORTS) stake = MAX_STAKE_LAMPORTS;
  return stake;
}

await runBot({
  onBatch(state, batch) {
    const slot = batch.slot;
    const { caps } = state.hello;
    const tx = state.txFeeLamports();
    const orders: OrderInput[] = [];
    let free = state.freeCashLamports();

    // 1) Learn from fills first: entries, exits, streaks.
    applyFills(batch, slot);

    // 2) Update per-mint windows from this batch's events, in chain order.
    for (const e of batch.events) {
      const t = trackFor(e.mint);
      if (e.event === 'CreateEvent') {
        // fresh curve; windows start empty
      } else if (e.event === 'TradeEvent') {
        // Per-trade price from the event's own post-trade reserves: the
        // state curve only holds the batch-end reserves, so using it here
        // would stamp every trade in a batch with the same price and hide
        // the dip-and-reclaim shape inside the batch.
        const vt = BigInt(e.virtual_token_reserves);
        const vq = BigInt(e.virtual_quote_reserves);
        const price = vt > 0n ? (vq * SCALE) / vt : 0n;
        if (price > 0n) {
          t.prices.push(price);
          if (t.prices.length > WINDOW_TRADES) t.prices.shift();
        }
        const sol = BigInt(e.sol_amount);
        if (e.is_buy) {
          t.buySol.push(sol);
          t.sellSol.push(0n);
        } else {
          t.buySol.push(0n);
          t.sellSol.push(sol);
        }
        if (t.buySol.length > WINDOW_TRADES) {
          t.buySol.shift();
          t.sellSol.shift();
        }
        t.trades += 1;
      } else if (e.event === 'CompleteEvent') {
        t.complete = true;
      }
    }

    // 3) Exits first: they free position slots for entries below.
    let freedSlots = 0;
    for (const [mint, pos] of [...openPos.entries()]) {
      if (pendingSells.has(mint)) continue; // sell already in flight
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue; // harness closes completes
      const holding = state.positions.get(mint);
      if (holding === undefined || holding.token_amount <= 0n) {
        openPos.delete(mint);
        continue;
      }
      const cur = priceOf(state, mint);
      if (cur <= 0n || pos.entryPrice <= 0n) continue;
      const retBps = ((cur - pos.entryPrice) * 10000n) / pos.entryPrice;
      if (retBps > pos.peakRetBps) pos.peakRetBps = retBps;
      const held = slot - pos.entrySlot;
      const hardStop = pos.sleeve === 'B' ? HARD_STOP_B_BPS : HARD_STOP_A_BPS;
      const shouldSell =
        retBps >= BigInt(TAKE_PROFIT_BPS) ||
        (pos.peakRetBps >= BigInt(TRAIL_ARM_BPS) &&
          retBps <= pos.peakRetBps - BigInt(TRAIL_GIVEBACK_BPS)) ||
        retBps <= -BigInt(hardStop) ||
        held >= MAX_HOLD_SLOTS ||
        (retBps <= -BigInt(BLEED_BPS) && held >= BLEED_SLOTS);
      if (!shouldSell) continue;
      if (orders.length >= caps.max_orders_per_batch) break;
      if (free < tx) break;
      const sell: SellInput = {
        side: 'sell',
        client_id: state.clientId(),
        mint,
        token_amount: holding.token_amount,
        max_slippage_bps: MAX_SLIPPAGE_BPS,
      };
      orders.push(sell);
      pendingSells.add(mint); // position stays tracked until the sell fill lands
      free -= tx;
      freedSlots += 1;
    }

    // 4) Entries: sleeve A cleanse + sleeve B recycle, in event order.
    let openSlots =
      caps.max_open_positions - state.committedPositionCount() + freedSlots;
    for (const e of batch.events) {
      if (orders.length >= caps.max_orders_per_batch) break;
      if (openSlots <= 0) break;
      if (e.event !== 'TradeEvent') continue;
      const mint = e.mint;
      const t = tracks.get(mint);
      if (t === undefined || t.complete || t.trades < MIN_TRADES_TO_ENTER) continue;
      if (openPos.has(mint)) continue; // already holding this mint
      if (pendingBuys.size > 0) {
        let pendingHere = false;
        for (const p of pendingBuys.values()) if (p.mint === mint) pendingHere = true;
        if (pendingHere) continue;
      }
      const curve = state.curves.get(mint);
      if (curve === undefined) continue;
      if (curve.real_quote_reserves < MIN_REAL_SOL_LAMPORTS) continue;
      if (curve.real_quote_reserves >= NEAR_COMPLETE_REAL_SOL) continue; // graduation-aware
      const cap = state.maxBuyLamports(mint);
      if (cap < MIN_BUY_LAMPORTS) continue;
      const buyShare = windowBuyShareBps(t);
      const { dipBps, reclaimBps, runupBps } = windowDipReclaim(t);
      const cur = priceOf(state, mint);
      if (cur <= 0n) continue;
      // Cleanse shape: a real run-up, then a flush, then a reclaim — with
      // fresh buy flow right now. A dip without the run-up is just a dying coin.
      const cleanseShape =
        runupBps >= BigInt(RUNUP_BPS) &&
        dipBps >= BigInt(DIP_BPS) &&
        reclaimBps >= BigInt(RECLAIM_BPS) &&
        cur > windowMeanPrice(t) &&
        recentBuys(t, 3) >= 2;

      let sleeve: 'A' | 'B' | null = null;
      if (!t.sleeveAEntered && buyShare >= BigInt(MIN_BUY_SHARE_BPS) && cleanseShape) {
        sleeve = 'A';
      } else if (
        !t.sleeveBEntered &&
        t.sleeveAEntered &&
        t.lastProfitExitSlot >= 0 &&
        slot - t.lastProfitExitSlot <= RECYCLE_WINDOW_SLOTS &&
        buyShare >= BigInt(RECYCLE_BUY_SHARE_BPS) &&
        cur > (t.lastProfitExitPrice * 102n) / 100n &&
        dipBps >= BigInt(RECYCLE_DIP_BPS) &&
        reclaimBps >= BigInt(RECLAIM_BPS)
      ) {
        sleeve = 'B';
      }
      if (sleeve === null) continue;

      let stake = stakeFor(state, buyShare, sleeve);
      if (stake > cap) stake = cap;
      if (stake < MIN_BUY_LAMPORTS) continue;
      if (free < stake + tx) continue;
      const clientId = state.clientId();
      const buy: BuyInput = {
        side: 'buy',
        client_id: clientId,
        mint,
        quote_lamports: stake,
        max_slippage_bps: MAX_SLIPPAGE_BPS,
      };
      orders.push(buy);
      pendingBuys.set(clientId, { mint, sleeve, slot });
      if (sleeve === 'A') t.sleeveAEntered = true;
      else t.sleeveBEntered = true;
      free -= stake + tx;
      openSlots -= 1;
    }

    return orders;
  },
});
