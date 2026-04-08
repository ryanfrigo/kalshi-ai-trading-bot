# Browser-Backed Ideas Feed Scraper with Trader Identity Resolution

## Summary
Build a browser-backed scraper for `https://kalshi.com/ideas/feed` that is read-only and feeds mirror discovery, not trading. It should extract each public trade post and then resolve the trader identity before anything goes downstream.

For trader discovery, use a two-stage policy:
- Preferred: explicit profile link or canonical handle from the feed row
- Fallback: fuzzy match the display name against an existing trader watchlist / known aliases when the feed does not expose a clean handle

Anything low-confidence stays as an unconfirmed candidate and does not auto-promote into the mirror set.

## Key Changes
- Add a browser-backed feed scraper worker.
  - Load the Ideas feed in a real browser session.
  - Extract visible public trade rows/cards.
  - Filter to trades over `$100`.
  - Normalize each row into a structured record: market, side, size, timestamp, visible trader text, and source URL.

- Add trader identity resolution.
  - If the feed exposes a profile link or canonical handle, use that as the trader identity.
  - If not, fall back to fuzzy matching on display name / alias / prior feed mentions.
  - Persist the identity result with a confidence score and evidence trail.
  - Keep explicit vs fuzzy matches distinguishable in storage and in the UI.

- Integrate into mirror discovery.
  - Feed-derived trader identities populate a candidate mirror watchlist.
  - The main bot can score those candidates for relevance and historical performance.
  - The scraper output remains advisory only; it does not trigger trades directly.

- Make it safe to run alongside the bot.
  - Put the scraper on its own cadence and TTL.
  - Cache results and back off on checkpoint / rate-limit responses.
  - If the feed is blocked or stale, the bot keeps running with the other signal sources.

- Use browser MCP for development validation only.
  - Use Chrome DevTools MCP to verify the live DOM, selectors, and rendered rows.
  - Do not depend on MCP at runtime.
  - If the page structure changes, fail closed and mark the source stale rather than guessing.

## Runtime / Integration Shape
- Input: public Ideas feed page rendered in a browser.
- Output: normalized feed items plus resolved trader identities.
- Downstream use:
  - mirror candidate discovery
  - trader relevance scoring
  - source-health reporting
- Identity policy:
  - explicit profile link / handle wins
  - fuzzy matching is allowed only as fallback
  - low-confidence identity stays unconfirmed until reinforced by later feed evidence

## Test Plan
- Browser validation:
  - use Chrome DevTools MCP to confirm the feed renders and selectors still match the DOM
  - verify rows expose trader text and, when available, a profile link/handle
- Unit tests:
  - parse and normalize feed rows
  - resolve explicit handles correctly
  - fuzzy-match display names when no handle is present
  - reject ambiguous or low-confidence identity matches
- Integration tests:
  - feed output flows into mirror discovery
  - scraper outages do not block trading
  - cached data is reused until TTL expires
- Main-bot tests:
  - live decision path still works when the Ideas feed is unavailable
  - feed-derived traders appear as advisory candidates only
  - no direct-trade coupling is introduced

## Assumptions
- The feed is publicly visible but checkpointed, so scraping must be browser-backed and conservative.
- The `Handle + fuzzy` policy is the right balance here: explicit identity when possible, fallback matching when needed, but no automatic promotion on weak evidence.
- Chrome DevTools MCP is for development and test verification, not the production runtime.
- If the feed structure changes, the scraper should fail closed and report stale/unusable rather than inventing trader identities.
