# From Accountability to Affordability: An Enterprise Guide to the Real Cost of Conversational AI Agents

Enterprises adopting conversational AI agents eventually run into the same question, right after the novelty of launch wears off: what does this actually cost to run, and how do you keep it under control as it scales? I've written before about [the case for adopting agents in the first place](https://www.linkedin.com/pulse/embracing-ai-agents-enterprise-guide-next-gen-voice-text-gorle-ay4jf) and about [measuring and governing them once they're live](https://www.linkedin.com/pulse/from-adoption-accountability-enterprise-guide-evaluating-gorle-4wavf) — worth a look if either is still open for you — but this piece stands on its own, and it tackles the question those two don't: cost.

If you've deployed a conversational agent, finance has probably already asked you this question, and you've probably answered it badly — not because you don't have numbers, but because "cost" for an agent isn't one number. It's a stack. Most cost conversations collapse that stack into a single line before anyone's actually priced it.

## Why cost isn't a launch-day decision

A GUI-era feature has a cost profile enterprises understand instinctively: build it once, and the marginal cost of one more user clicking a button is close to zero. A conversational agent doesn't work that way. Every single turn — every sentence a user says and every reply the agent gives back — touches metered services that charge per minute, per token, or per character. Cost doesn't sit still after launch; it scales with usage the way infrastructure does, and it can move even when your product hasn't changed at all, because the providers underneath it change their pricing, or your usage mix shifts.

That means the same discipline that keeps an agent accurate and safe — measure continuously, don't assume launch-day numbers still hold — applies just as much to cost. The teams that get surprised by their AI bill are usually the ones who priced it once, at launch, and never looked again.

## 1. Cost isn't one number, it's a stack

This looks different depending on the mode. For a voice conversation, a single turn touches at least three metered services before you even count the channel it's delivered over. For a text-based conversation — say, over WhatsApp or RCS — drop the STT and TTS legs entirely; you're left with just the LLM and the channel. Voice is the fuller stack, so it's the one worth walking through in full:

**STT / ASR (Speech-to-Text, also called Automatic Speech Recognition)** converts what the user said into text the LLM can reason over. Pricing is pegged to audio duration, but the billing granularity itself varies by provider — Sarvam, for instance, bills to the second (quoted as a per-hour rate), while others meter in per-minute blocks — and providers differ meaningfully on both price and accuracy — especially on accented or code-switched speech. That accuracy gap is a real cost in itself: a misheard word sends a wrong transcript to the LLM, which is now money spent reasoning about the wrong thing.

**The LLM (the reasoning core)** is priced per input and output token — and, counterintuitively, it's often the cheapest line in the stack for a short, transactional exchange, not the most expensive. A model can decide and reply in a few hundred tokens; the audio wrapped around that reply doesn't get any cheaper just because the reasoning was quick. Hybrid model routing — a lightweight, fast model for routine intent classification, a heavier reasoning model only when the conversation genuinely needs it — still matters for latency and for reasoning-heavy scenarios, but don't reach for it expecting it to be your biggest lever on cost; for most voice bookings, it isn't.

**TTS (Text-to-Speech)** converts the agent's reply back into audio, priced per character generated — nearly every provider works this way, from the majors (Google, Azure, Amazon Polly, OpenAI, ElevenLabs) down to the ones in this stack (Deepgram, Groq Orpheus, Sarvam). That's a useful contrast with STT: the input leg is priced by audio duration, the output leg by text length, so a wordier reply costs more regardless of how quickly it's spoken. In practice, this is the leg that tends to surprise people: it's billed on every character of the reply regardless of how terse the underlying reasoning was, and in short voice exchanges it's routinely the largest line in the stack — not the LLM. Voice quality and latency vary here as much as price does, and for regional-language deployments your provider options narrow considerably before cost even becomes the deciding factor.

**Channel and delivery** sit underneath all of it. Voice telephony is usually billed on a pulse — often 30-second or 60-second increments, rounded up — rather than metered continuously to the second; Exotel, for instance, bills on a pulse rate on top of its credit-based plan structure. WhatsApp moved off conversation-based pricing in mid-2025 and now bills per message, split by category — marketing, utility, and authentication messages each carry their own per-message rate, while service replies inside the customer-initiated 24-hour window are free. RCS is also priced per message, tiered by message type — a plain text message, a richer single message with buttons or images, and a session-based "conversational" message can all cost different amounts. Each channel has its own rate card, and often its own regulatory and delivery-reliability quirks on top of price.

None of these four move together. A provider that's cheap on LLM tokens can be expensive on TTS, or fast on one leg and slow enough on another that you quietly lose the conversion you built the agent to protect.

I ran this exact comparison on my own playground, using the same appointment-booking scenario on two very different stacks — GPT-5 mini with Deepgram's STT and TTS, and Sarvam's models end to end. In both runs, TTS was the single largest line, running roughly 5x the LLM's cost on the first stack and nearly 20x on the second; STT came in second both times. The LLM — the leg every cost conversation seems to start with — was the cheapest line in the stack, in both runs. That's not a universal law; a longer agentic loop or a reasoning-heavy scenario would likely flip it. But for a short, transactional voice booking, "AI is expensive because of the model" turned out to be the wrong instinct.

I also ran the same booking as a text-only exchange — no STT, no TTS, just the LLM and the channel — comparing GPT-5 mini against Sarvam 30B. Sarvam used more tokens to get through the conversation, not fewer, and still came out roughly 6x cheaper: token count and price-per-token are two different levers, and only one of them shows up on the invoice.

**[FIGURE 1 — "The Conversational Agent Cost Stack" diagram]**

## 2. The metric that actually matters: cost per resolved task

"Latency and cost per resolved task" deserves its own explanation, because it's the one number that lets you compare configurations honestly — and it's rarely the number a cost conversation starts with.

Cost per call, cost per minute, cost per token — these are all inputs, not answers. None of them tell you whether you got your money's worth. A cheaper model that needs three clarifying turns and still escalates to a human costs more, end to end, than a pricier model that resolves the same task in one exchange. Optimize any single leg of the stack in isolation, and you can end up with an agent that's technically cheaper per API call and more expensive per outcome — a false economy that looks great on a unit-cost slide and terrible on a P&L.

Cost per resolved task forces the stack and the outcome into the same number. It ties your business metrics — did the task actually get done, was the customer satisfied — to your agent's own performance: correctness determines how many turns and retries you pay for; the unit economics of the stack determine what each of those turns actually costs. You can't govern one without the other, and you shouldn't try to optimize one without the other either.

**[FIGURE 2 — "Cheap Per Call ≠ Cheap Per Outcome" diagram]**

## 3. Costs drift just like accuracy does

Agents don't just fail silently over time — they get quietly more or less expensive, too, and for reasons that have nothing to do with anything your team changed. Model providers revise pricing. Usage mix shifts toward voice from text, or toward a language your STT and TTS providers price differently. A provider improves quality without changing price, which moves your effective cost per resolved task even though your bill per API call looks identical.

Treat cost the way you'd treat any other quality metric: re-baseline it every time you change a model, a prompt, or your scenario mix — not once a quarter when finance happens to ask. Cost drift is a leading indicator the same way a rising hallucination rate is; by the time it shows up as a surprising invoice, it's been building for weeks.

## 4. Building a cost instrumentation layer

The practical fix is unglamorous: log cost per session broken down by leg — STT, LLM, TTS, channel/delivery — not just as a total. Track effective cost per minute alongside cost per resolved task, and if you serve markets where finance budgets in a currency other than USD, track cost natively in that currency rather than converting list-rate USD after the fact; exchange-rate noise makes trend lines useless otherwise.

I built a small internal playground for exactly this reason — a tool to swap LLM, STT, and TTS combinations against the same scenario (a hospital appointment booking flow, in my case, given where most of my own agent work has lived) and watch cost, latency, and token usage update live, in USD or INR, per leg of the stack. It's a prototype, not a production system, but the comparison earlier in this piece — TTS and STT outweighing the LLM by 5x to nearly 20x — came straight out of it. The exercise of watching those cost lines move independently, sometimes in opposite directions when you swap a single provider, is the fastest way I've found to build real intuition for where an agent's money actually goes, instead of guessing from a vendor's list price.

**[FIGURE 3 — "Real Cost, Real Runs" data graphic: voice-mode STT/LLM/TTS breakdown for two stacks, plus the text-only LLM comparison]**

## The bigger point

Conversational agents didn't just change the interface between users and software; they changed the finance conversation that comes with it. GUI-era software had a cost profile that was mostly fixed after launch. Agents meter every interaction, across four independently-priced legs — and the fourth one changes shape entirely depending on whether you're on voice, WhatsApp, or RCS — and that cost moves whether or not you're watching it.

Enterprises that treat agent cost as a one-time procurement decision — priced at launch, filed away, revisited only when the bill looks wrong — will find it drifting away from them quietly, the same way unmonitored accuracy does. The ones that build a cost instrumentation habit alongside their evaluation habit will be the ones who can actually answer finance's question next quarter, with a real number instead of a shrug.

I hope this helps businesses looking for a guide to the cost side of conversational AI agents. If you want to see the stack broken down for yourself, I built a small [cost-comparison playground](https://35-234-215-193.sslip.io/) alongside this piece that you're welcome to try.
