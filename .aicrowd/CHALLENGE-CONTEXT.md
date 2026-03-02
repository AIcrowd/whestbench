# Challenge Call: AIcrowd Challenge Proposal

## Challenge Title

**Mechanistic Estimation for Boolean Circuits**

---

## Organization

**Alignment Research Center (ARC)**
Non-profit research organization conducting theoretical research aimed at aligning future machine learning systems with human interests.
Website: [https://www.alignment.org/](https://www.alignment.org/)

---

## Contact Information

* **Contact Name:** Jacob Hilton
* **Email:** [jacob@alignment.org](mailto:jacob@alignment.org)
* **Phone:** 415-230-9976

---

## Administrative Details

* **Created At:** February 10, 2026
* **Updated At:** February 10, 2026

---

## Challenge Description

This is a theoretical computer science challenge from the Alignment Research Center (ARC). It is based on the idea of producing **"mechanistic estimates"** for the loss of a neural network, as discussed in the following blog posts:

* [https://www.alignment.org/blog/competing-with-sampling/](https://www.alignment.org/blog/competing-with-sampling/)
* [https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

The challenge will involve **Boolean circuits** instead of neural networks.

Participants will design an algorithm that:

1. Accepts a description of a Boolean circuit.
2. Outputs the expected value of the output of the circuit (treating `TRUE` as 1 and `FALSE` as 0) on a uniformly random input.

The circuits will be generated randomly in a specific way designed to produce **deep circuits with complex behavior**.

---

## Motivation

We are excited to run this challenge to spur research on mechanistic estimation.

Boolean circuits provide a strong testing ground where cleverly designed mechanistic estimates can perform well, and where there is rich structural behavior to exploit.

The natural baseline algorithm is to:

* Run the circuit many times on random inputs.
* Take the empirical expectation.

However, we have already designed simple mechanistic algorithms (as yet unpublished) that outperform this baseline.

We expect participants to:

* Rediscover these approaches.
* Explore a wide space of more sophisticated ideas.
* Help convey our research to a wider audience.
* Develop novel techniques of direct interest to our research agenda.

---

## Timeline

We will provide:

* A description of the procedure used to generate circuits.
* Potentially, a dataset of example circuits generated using this procedure.

Participants will also be able to generate unlimited data themselves using the provided generation procedure.

Participants must submit an algorithm that estimates the expected value of the circuit on a uniformly random input.

---

## Evaluation Criteria

Submissions will be evaluated on held-out circuits using:

* **Mean Squared Error (MSE)**
* Subject to a **runtime constraint** on the algorithm

Evaluation may include:

* **In-distribution evaluation** — circuits generated using the same method.
* **Out-of-distribution evaluation** — circuits generated using variant methods.

---

## Organizers Bio

We are a non-profit organization founded in 2021.

Our research has been published at top machine learning and theoretical computer science conferences.

We have not previously run a public challenge.

---

## Prize & Logistics

* **Prize Pool:** $50,000 – $100,000
* Philanthropic funding secured to cover:

  * Prize money
  * Operational costs associated with running the contest

We would love to hop on a call to discuss further.



# Email Thread: AIcrowd × Alignment Research Center

**Subject:** Challenge Call Response – Mechanistic Estimation for Boolean Circuits
**Participants:** Sharada Mohanty (AIcrowd), Jacob Hilton (Alignment Research Center)
**Date Range:** Feb 10–12, 2026

---

# 📩 Full Email Thread (Verbatim Content Preserved)

---

## 📨 Email 1 — Challenge Call Submission

**From:** [no-reply@aicrowd.com](mailto:no-reply@aicrowd.com)
**To:** Sharada Mohanty
**Date:** Feb 10, 2026
**Subject:** New Challenge Call Submission

---

### Challenge Call Details

* **Call:** Challenges on AIcrowd
* **Contact Name:** Jacob Hilton
* **Organization:** Alignment Research Center
* **Phone:** 415-230-9976
* **Email:** [jacob@alignment.org](mailto:jacob@alignment.org)

---

### Proposal Title

**Mechanistic Estimation for Boolean Circuits**

---

### Proposal Summary (Exact Submission Text)

> This is a theoretical computer science challenge from the Alignment Research Center ([https://www.alignment.org/](https://www.alignment.org/)), a non-profit research organization conducting theoretical research aiming to align future machine learning systems with human interests.
>
> The challenge builds on the idea of producing "mechanistic estimates" for neural network loss, as discussed in:
>
> * [https://www.alignment.org/blog/competing-with-sampling/](https://www.alignment.org/blog/competing-with-sampling/)
> * [https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)
>
> Instead of neural networks, the challenge focuses on Boolean circuits.
>
> Participants must accept a description of a Boolean circuit and output the expected value of the circuit output (TRUE = 1, FALSE = 0), assuming uniformly random inputs.
>
> Circuits are randomly generated in a structured way to produce deep circuits with complex behavior.

---

## 📨 Email 2 — Sharada’s Initial Response

**From:** Sharada Mohanty
**To:** Jacob Hilton
**Date:** Feb 10, 2026

---

> Dear Jacob,
>
> Thank you for your interest in AIcrowd, and for the very interesting challenge proposal.
>
> It is also great to hear from you, as this closely aligns with a direction I have personally been very interested in — developing a series of meaningful, well-designed challenges around alignment and mechanistic interpretability on AIcrowd.
>
> Would you be available for a short call later this week (or early next week), for us to get a better understanding of the task and to discuss next steps?
>
> Before we speak, it would be helpful to clarify a few points:
>
> 1. Circuit Difficulty
>
>    * Do you already have a working notion of circuit difficulty?
>    * Is difficulty tied to depth, gate reuse, gate types?
>    * Does difficulty influence test set stratification?
> 2. Evaluation Design
>
>    * Runtime constraint + MSE on held-out circuits
>    * Should runtime budgets be per-circuit or per-batch?
>    * Should resource usage (CPU, memory) influence scoring?
> 3. Algorithm Scope
>
>    * Are per-instance reasoning methods preferred?
>    * Are offline-trained predictors acceptable?
>    * Should lightweight code review (e.g., LLM-as-judge) be considered?
> 4. Adversarial Optimization / Leaderboard Gaming
>
>    * Fitting generator parameters instead of solving circuits
>    * Embedding offline-trained predictors
>    * Exploiting floating-point quirks
>    * Guardrails to prevent unintended optimizations?
> 5. Timeline
>
>    * Potential NeurIPS Competition Track alignment?
>    * Earlier launch possible?
> 6. Blue Sky Award (Optional)
>
>    * Additional prize for extraordinary performance beyond expectations?
>
> Optional: If available, a minimal end-to-end reference implementation would help build shared intuition ahead of the call.
>
> Best,
> Mohanty

---

## 📨 Email 3 — Jacob’s Technical Clarifications

**From:** Jacob Hilton
**To:** Sharada Mohanty
**Date:** Feb 12, 2026

---

### Circuit Construction (As Described in Email)

> Proposed generator:
>
> * Start with n input wires (uniform random inputs)
> * For each of L layers:
>
>   * Replace n wires with new n wires
>   * Randomly pair wires
>   * Connect each pair with one of 16 possible 2-input gates
> * Final output: n output wires
>
> Goal: Estimate probability each output wire is on.
>
> Parameters:
>
> * n (width), e.g., 1024
> * L (depth), 1 to 256
> * Computational budget
> * Target MSE: 1e-3 to 1e-6

---

### Evaluation Philosophy (Exact Points Raised)

> * Constrain CPU and memory per circuit (per-batch possible)
> * Algorithms should take a resource budget as input and adapt
> * Performance improves sharply with compute
>
> Ideal properties:
>
> 1. Consistent computational environment
> 2. Transparent runtime feedback
> 3. ±10% compute buffer with linear score adjustment
> 4. CPU usage should factor into score (not just runtime cap)

---

### Algorithm Scope

> * Open to broad approaches
> * No LLM-based code review planned
> * Will red-team further before launch

---

### Adversarial Considerations

> * Expect best methods to be mechanistic
> * May run a trial hackathon to stress-test design

---

### Timeline

> * Desire to launch ASAP
> * Interested in compatibility with NeurIPS timeline

---

### Code Access

> Shared minimal repo:
> [https://github.com/alignment-research-center/circuit-estimation-mvp](https://github.com/alignment-research-center/circuit-estimation-mvp)

---

## 📨 Email 4 — Sharada’s Follow-Up & Platform Considerations

---

### Call Scheduling

> Proposed slot: Monday, 16th 9:00–10:00 PT

---

### Infrastructure & Evaluation Design Discussion

> Resource Enforcement Options:
>
> * Soft enforcement: Penalize overages
> * Hard enforcement: Throttle processes (container-level limits)
>
> Runtime Budget Adaptivity:
>
> * Reference: Data Purchasing Challenge (2022)
> * Participants often implemented regime-specific if-else logic rather than true adaptivity
> * Need careful budget exposure to incentivize genuine adaptive algorithms
>
> CPU Usage in Score:
>
> * Avoid unintended leaderboard pathologies
> * Avoid distorted optimization dynamics
>
> Environment Consistency:
>
> * All submissions run on predefined AWS instances
> * Detailed runtime and resource dashboards available
> * Participants can spin up identical AWS instances for validation

---

### Red-Teaming Proposal

> * Launch unlisted invite-only beta
> * Stress-test evaluation protocol
> * Validate leaderboard dynamics before public release

---

### Timeline Strategy Proposal

> Infrastructure readiness estimate: ~4 weeks + beta phase
>
> NeurIPS considerations:
>
> * Proposal deadline: mid-April
> * Acceptance: mid-May
> * Competition window: late June–October
>
> Suggested strategy:
>
> 1. Launch when ready
> 2. Submit to NeurIPS in parallel
> 3. If accepted, introduce additional NeurIPS-affiliated phase

---

## 📨 Final Acknowledgment

> Sharada acknowledged receipt of repo and planned deeper review ahead of call.

---

# End of Thread


-----

# Feb 16, 10:32 PM

**Meeting Date:** 16th Feb, 2026 - 10:32 PM

---

**Jacob Hilton** *[00:00]*: Or something, maybe. Paul, do you want to leave this? 
**Paul Christiano** *[00:03]*: I'm happy to. Definitely. Also happy for you too. We should also talk about timelines at some point and like how it fits into ARC's near term strategy. I think there's a couple options there and maybe missed this in the email thread, but I don't have a good sense of what timelines are realistic on your side and what we should be targeting as my super short description for what we're looking for is we are interested in algorithms that rather than estimating things by just running a bunch of samples through them, analyze the structure of some circuit or some neural network to make a good estimate. We think we've done some preliminary work and think that it's a pretty rich problem. Like there is a lot of room for incrementally better algorithms that improve on it and a fairly challenging problem at least. 
**Paul Christiano** *[00:47]*: Like we, in a month of effort probably can't get close to the best possible performance. So we are very excited to have a contest to see what people are able to come up with this problem. The circuit estimation problem is one that we might work on or might not work on. So I think if we ran a contest we would delay working on it ourselves until we saw what happened. We've done a little bit of work on it, like probably spent a few weeks, I would say thinking about it so that we know that it's a real problem and is interesting and there's not something easy to do, but would be running a contest prior to us doing a deep dive or writing a paper that really digs in on really great algorithms for it. 
**Paul Christiano** *[01:29]*: Though we may be writing a paper in early April that covers some very simple algorithms for this problem, which would be like a reasonable baselines. I don't know if I actually hit the most important parts, but that's what's on my mind about this. 
**Mohanty** *[01:47]*: Yeah, that's useful, right? So that's already a good foundation to kind of stand upon. And then I think the good part is you already have some initial exploration around the problem, so you have a rough idea of the complexity of the solution space. And now basically in terms of the broad timelines and before we jump into the other technicals, I think based on initial assessment, we feel comfortable to be able to roll out a public challenge in about three to four weeks, one where we have kind of done everything that's needed in terms of preparing the starter kits, including also having some well documented baselines and having all the design communication and all that stuff sorted here. I think Most of the work would go into ensuring how the starter kit can make the problem accessible for the broader community. 
**Mohanty** *[02:33]*: And there also I think it shouldn't be like that hard. Right? And once that is done, just to connect back with the stuff were talking about with the NeurIPS timelines, etc. Again, if the whole goal is to run the challenge as soon as we can, the Neurips timelines won't work because again we are submitting a proposal in April and then getting accepted in May to run a competition in June and stuff that won't work. But the same thing that I mentioned on the email thread, we can still run the challenge and in parallel basically just submit it to New Ribs, the competition track and if it works out then basically we add an extension of a round saying that oh, now that it's a part of Neurips, the final prizes would be given out after the end of the next round. 
**Mohanty** *[03:17]*: And there we can basically distill whatever learnings we have from the first round or the first few rounds that we have run until then. And running the first big boss mode of the challenge at Neurips. 
**Paul Christiano** *[03:31]*: When do the NERPS contests start? Do they actually start in like mid June or something? 
**Mohanty** *[03:35]*: Yeah, officially. Yeah, officially they're supposed to start around mid June and they go on to like mid October, end of October to give a month of breathing room for all the logistics of inviting participants etc. But in principle you can run them earlier. Right? Many of the Neurips challenges we ran we would run like a pre Neurips version of it to get a sense of the difficulty of the problem and then just run an extra round. 
**Paul Christiano** *[04:01]*: And there are various options like that. We could do like a circuits version or some TCS flavored version in advance and then for Neurops run like a neural net version or something like that. I don't know exactly what a Neurops audience would be. I don't know what the upside of that is. I don't know if it's something we should seriously consider. 
**Mohanty** *[04:22]*: Yeah, I think we shouldn't make that decision already where right now we just don't have enough information to make an informed decision there. I would say let's focus on the version of the challenge that you have deeply thought about and there it's a lot more risk averse to kind of run it as is. But we try to do the best job we can at running it and really driving engagement in the problem. And along the way we learn a lot of things right on how the community responds to this and a more empirical idea of the solution landscape. And in parallel, again, we just try and see, okay, how best to kind of document a baseline or whatever the competition results are. And we use the new risk proposal as an excuse to do it. 
**Mohanty** *[04:59]*: And if in April your paper comes out, then even if the challenge is already running right. Then we basically add it, add the release paper as like a resource for the challenge for participants to say, oh look, here's some documentation on some more baselines that are released by arc. 
**Paul Christiano** *[05:16]*: My guess, I'm interested in Jacob's opinion on this. My guess is that we don't want to launch a challenge until like the couple weeks, like until the paper is out or the couple weeks after that. 
**Mohanty** *[05:24]*: Okay. 
**Paul Christiano** *[05:25]*: So I think our plan would be we're releasing a pair of papers on more like TCSE settings and ML settings or on like and then also want to do a bunch of like public communication and outreach around that month. And so starting up, I think we think it would be nice to be starting up a contest around the same time. So one of the things we're talking about is here are results, here's motivation and why this stuff matters. And also here's a contest that is starting next week or whatever. 
**Mohanty** *[05:52]*: Okay. 
**Paul Christiano** *[05:53]*: And that's roughly how were imagining it. So that would be a reasonably long, I mean, I guess it's not that long a timeline, but that would give us like a six week or. Yeah, that's eight week timeline to spend. 
**Mohanty** *[06:01]*: That's perfect breathing room. And there also we have enough time to also do the beta test and the red teaming that we wanted to do anyway. So like we should still aim for kind of really having a version that we are proud of in releasing in closed circles in like three to four weeks and then have two to three weeks of breathing room for whatever we learn from the effort to kind of really put it back in the challenge. 
**Jacob Hilton** *[06:28]*: I would imagine. Yeah. Realistically we would probably be putting out papers mid April probably. If we have initial version done by end of March, then having a few weeks to red team then would be good. 
**Mohanty** *[06:42]*: Okay. 
**Jacob Hilton** *[06:43]*: Yeah, the quicker the, you know, the faster we get stuff done now, the more breathing room we'll have. 
**Mohanty** *[06:49]*: Yeah, I think then I think the neerips timelines could work. Right. Because I initially thought that you really wanted to kind of launch it in like within the next three, four weeks. And I was also internally checking out the logistics of how to make that happen. So this way again the paper comes out, you do your, the PR and the communications around it. And then we say, oh, by the way, there's a challenge happening around this. Then in parallel you can basically reuse a lot of the content from the paper in writing a compact Neurips competition proposal around it. Right. Because we have all the elements in place. 
**Paul Christiano** *[07:18]*: How much participation do Neurips competitions drive? Like what exactly is the size of upside from a Neurips competition? 
**Mohanty** *[07:24]*: It very much depends across competitions. The ones that we host, they usually have something between 300 to 500 participants over the six months that we run it for. The number of submission also varies, but then I would expect something like 5,000 to 8,000 ish submissions here. I think I would actually expect on the higher end of this distribution because again, many of the other neerips, like many of the current modern day challenges, they are like more LLME challenges where suddenly they have to kind of fine tune 70B model or like a 10B model and submit it here. Again, at least that burden is a. 
**Jacob Hilton** *[08:00]*: Lot less accessible challenge. 
**Mohanty** *[08:02]*: Yeah, but yeah, I think we should prepare for something in that order. Like at least like a 5,000 submissions and about 300 to 500 participants. 
**Paul Christiano** *[08:12]*: And I guess, yeah, being a NERPS challenge does significantly increase the perceived prestige from winning. It just does drive many more participants. Is that the observation? 
**Mohanty** *[08:21]*: Yeah, the observation is that again in our case we anyway managed to kind of gather a lot of participants, but the quality of participation we get is definitely a lot higher in context of the Neurops challenges. So the percentage of people in this distribution of participants who are like either already serious academics or seriously who want to pursue research that is a lot higher. And not just people who want to kind of ride the wave and just make their resume a little bit better by saying, look, I got so and so rank in this competition and I guess here again it's almost a field building exercise of some sorts. So you probably want more people who are fundamentally motivated by the research than just the competitive participants crowd. 
**Paul Christiano** *[09:07]*: Yeah, I think we do have a somewhat unusual social situation in that like for our perspective, the best upside is there's a smaller set of people who are going to be very excited about our research direction. Like best upside is probably hires. The thing we most want is there are people who get really excited about the problem, find us a good entry point and then are excited to work with us on similar problems in the future and it will probably very accessible. But also like it is not the case that people can just be doing this kind of challenge and then stop In. In the way that it might be for lots of, like, get a good performance on a benchmark. Tasks where they're kind of similar to tasks people would have been doing otherwise or tasks they're familiar with. 
**Paul Christiano** *[09:46]*: I don't know if that changes our calculus around timing NERPS competition. I guess there's other aspects we should also talk about, I guess, like what is the size of prizes and what is the prize structure? I think there are a bunch of options we had considered or been thinking about. There's prizes for best performance as the bulk of prizes. At the end of the period, we sort of wanted a prize for the best algorithmic contribution, someone who's actually written up the algorithm clearly. And we think it is a good. If you sort of abstract out the performance optimization, which is actually the best algorithm. I had some interest in having intermediate waypoints, like one third of the way or 1/4 of the way through. You're like, well, here's the best algorithm that's been submitted so far. 
**Paul Christiano** *[10:31]*: And they also get some prize to incentivize more activity earlier. But I don't know if that makes sense. 
**Mohanty** *[10:35]*: Yeah, split the competition into multiple rounds. So we also have a little mechanism in which we keep distilling the learnings from running it for a month and a half into the next round. So there you could just announce a prize for say, round one and then round two, so on and so forth. 
**Jacob Hilton** *[10:54]*: Do you normally change the, like the evaluation criteria from round to make it a bit more interesting? Like, would we want to like increase the size of the circuits or something like that, or. I don't know. 
**Mohanty** *[11:08]*: Yeah, we have run some challenges where the specs of the benchmark won't particularly change across rounds, including the recent chess one. But in many, we definitely want to kind of have an added. An increased level of complexity on the problem across the rounds. Or in some cases we have completely changed it where basically for completely new problems, we would design round one as kind of a gateway drug of sorts, where it's a very simple formulation, we know everyone will do it well, but it gives us a good sense of how many people manage to actually get a score above a particular threshold. And then in round two, we actually introduce the concrete problem that we want them to focus on. 
**Jacob Hilton** *[11:48]*: Right, Right. From that perspective, let's say we did want to go for the Neurips submission. Is having an earlier round kind of a benefit or is it a cost? Because then people are less interested in the actual main event. 
**Mohanty** *[12:02]*: If we have clarity on how The Neurips version of the challenge basically incrementally adds to what we already have. It ends up becoming a benefit because then we are not saying, look, here we are a bunch of people who want to run a challenge, but here we are a bunch of people who have hands on experience at actually running this challenge and we deeply understand the dynamics of how the community responds to this and also a much better understanding of the solution space, which allows us to kind of iterate a little bit more on figuring out how to increase the complexity of the problem. Paul, I think you're muted. 
**Paul Christiano** *[12:36]*: Yeah. When we're actually submitting, the Neurips timeline would be submit in April, learn in May, run in June. So we would at the time of submission not have run around. Yeah, I guess we could say that by the time we're running, we will have run around. 
**Mohanty** *[12:49]*: Actually. You're right. So usually again we have already run it from like Jan, Feb, et cetera, but there we say we are anyway going to run this round and by the time we basically run the narib's challenge, we would already have one round of this challenge ongoing. But again, this is like I have both reviewed Narish proposals and run a bunch of them. This is a really strong one. 
**Paul Christiano** *[13:10]*: Right. 
**Mohanty** *[13:11]*: If we just execute it. Well, I think I would be surprised if it doesn't get accepted. 
**Paul Christiano** *[13:17]*: A question I had is if I imagine a Neurips contest, I am more inclined to go the. So we have both papers, one on stuff like circuits and one on random MLPs like neural nets and initialization. I don't know if that is a real thing for the Neurips audience, but my inclination would have been do like a round on a more TCSE thing in advance and then make the NERPS contest and around for a more clearly machine learningy thing. Just because there might be some culture mismatch. If we're like here's a Turing Machine or a Boolean circuit or whatever, that's not as much what NERPS people care about. 
**Jacob Hilton** *[13:50]*: What do you think we would do? Do random MLPs or change it slightly? 
**Paul Christiano** *[13:53]*: Probably some kind of. Or maybe random RNNs or something like that. If we do some stress testing so we have code for random MLPs and we understand it reasonably well. Beating sampling is hard in deep MLPs. I mean, I'm also somewhat interested in that. 
**Jacob Hilton** *[14:11]*: I see. We could just take one of the regimes which we know we struggle with. 
**Paul Christiano** *[14:18]*: We're so bad, it's so hard to do we think it's probably possible like depth 20 should be fine or arbitrary depth, you should still be able to beat sampling. But I guess Mike's concern is that's kind of grungy. Like you're going to then do the low rank stuff and you're going to improve the optimization of the non linearity handling and stuff. I don't know how much people like optimization is crunchy. 
**Mohanty** *[14:38]*: I think you know, there I would still. The way I would think of it is basically here you have this unique way where there's a clear calibration of the problem difficulty. Right. Even in the MLP one that you were talking about. So just by changing the depth per se, you already can calibrate the difficulty. So the test set itself would be stratified. So what happens is when we run the challenge then basically again and we can evaluate it across different depths, then we could basically go ahead and say here's like the blue sky reward or something like that for the edge of the problem. That we don't know exactly if answer exists or how well we will get a result through the end of the competition. But even in other cases, the benchmark won't be saturated just because the problem is too hard. 
**Mohanty** *[15:22]*: We will still be able to get a signal to be able to create that gradient of performance between the submissions. As long as we have that notion of stratification, then it's okay to push it a little bit so that, you know, the whole community already knows that at the end of the competition. The whole thing is not solved per se, but we did have a clear understanding of how much progress we made. 
**Paul Christiano** *[15:43]*: So I guess this is a dimension to think about. Yeah, I think of doing a NERFS competition, it does feel appealing to do two rounds. I think we should think also about how much work that is and whether that's more work than we want to put in. But it does feel appealing to do kind of two different topics. I think there's a lot of topics we could do this on. 
**Jacob Hilton** *[16:01]*: Yeah, it's almost a bit of a shame to have two competitions and do them both on Circus. You know when like. 
**Paul Christiano** *[16:08]*: Yeah, that's kind of a random topic. 
**Jacob Hilton** *[16:10]*: Yeah, yeah. 
**Paul Christiano** *[16:11]*: I think the big things we should think internally, like probably RNNs is more interesting. Like if we set up an RNN domain or something. I don't know, we should figure out what is a. What is an interesting setting that feels maybe, or maybe MLPs is fine. We'll have this paper on MLPs where we present some baseline algorithms which be sampling at low depths, but don't be baseline. 
**Jacob Hilton** *[16:33]*: They're pretty intense. 
**Paul Christiano** *[16:34]*: Algorithms. Algorithms, yeah. I guess we would publish code presumably also for that. I don't know if we're planning to publish code in the paper with the paper. 
**Jacob Hilton** *[16:44]*: Yeah, I think so. 
**Paul Christiano** *[16:45]*: Yeah. So we publish code for implementations. 
**Mohanty** *[16:49]*: By the way, another aspect to think about this when you're talking about completely two different types of models to focus on across the rounds or different flavors of the competition, is that I would see the series of competitions as a community building exercise, right. Where the competition itself doesn't have an identity, but these different rounds and whatnot. The goal is to create a community and then we hope that people are already interested enough in the first round that a subset of them are super curious to see how much of whatever they explored and learned, especially in this earlier problem, translates into the new one. So it's perfectly fine to completely change the regime and definition of the problem. 
**Paul Christiano** *[17:24]*: The algorithms are very. I mean there are many twists, like you can't apply the same algorithms, but there is much, at least for the kinds of algorithms we're thinking about, which is maybe not optimal algorithms. There's a lot of overlap between the approaches. There probably is fundamentally like they're just very similar settings. So I guess I'm curious what mechanics there are to what are the important things to talk about mechanically about running this, like what work do we have. 
**Jacob Hilton** *[17:50]*: To do. 
**Paul Christiano** *[17:53]*: And what decisions do we need to make? Like where it sounds like you guys are maybe pretty sold that it would be a good competition to do and submit to neurips, but if there's other things that go into that. I don't know if you've actually made a call. 
**Mohanty** *[18:03]*: Yeah, exactly. So I did went through the technical details, many of the things that we have talked about, but the way this would work is we will first agree on some high level technical details like the ones we have already kind of spoken about. Right. Like for example, what kind of enforcement we want to do and how exactly are we going to specify the resource constraints? Is it like per circuit level, per batch level or like per evaluation level and etc. Based on that, we just agree on a specific framework and an interface and then what you would have to really prepare across just the round one specification of the problem, multi round one is exactly a repo that you have already created, but then a lot more well documented, etc. 
**Mohanty** *[18:44]*: Where we can take that and ensure that the interfaces are something that are well aligned with our evaluation setup. And we focus on in that repo on how we can just run it locally. That's it. Where you don't have to worry about the security of it. And then how do we impose the resource constraints, etc. But then we at our end translate that repo into something that can actually run at scale, where all the security aspects, etc. Are taken care about. We take care about how adversarial participants might react, how do we run it in a sandbox environment. All of that basically happens directly on our evaluation framework. While we do that, then basically the collaboration with you would just focus around ensuring this repo is always up to date or multiple versions of it across rounds and multiple up to date. 
**Mohanty** *[19:29]*: And we will do variations of this repo as like the official starter kit for the challenge as well, where we'll also inject some of our scripts, etc. To ensure people can run things locally, etc. And also probably reorganize the code documentation, etc. And then like the actual execution of it, right. When respect to the number of submissions that come in, etc. How do we actually run it? How do we provide support to participants? All of that basically can be handled at our end. Baselines again is something that of course the baseline that we will release, it'd be good if they come from you, but in many cases we also basically someone just gives us a high level spec of the baseline. We at least have someone from our research team quickly go implement something and release it. 
**Mohanty** *[20:10]*: But I guess given you're working on this, I'm assuming you already have some baselines that can be rolled out. 
**Paul Christiano** *[20:15]*: Yeah. One concern I have about baselining is I am concerned about participants having to spend a bunch of time on really annoying performance engineering right now. We have this NUMPY implementation for circuits, say. I think we have less confidence in implementation quality for circuits than for MLPs. For being performant, we have this NUMPY implementation. I'm concerned that it's five times slower than a performant C implementation would be or even an optimal NUMPY implementation. 
**Jacob Hilton** *[20:44]*: Should we. 
**Paul Christiano** *[20:44]*: I guess I would like to have a baseline that is where the forward pass is about as fast as it can be and where meanprop and covprop are about as fast. 
**Jacob Hilton** *[20:52]*: Should we just impose flop? A FLOP constraint instead of a CPU constraint? I mean, it's a bit weird for circuits, I guess, when they're not even. 
**Paul Christiano** *[21:03]*: Yeah, there's so many other operations you do and like the memory access pattern stuff is a big Deal. 
**Mohanty** *[21:09]*: Yeah, it will get a little bit tricky with the FLOP constraint, but let me think about it a little bit more. In principle also from a technical point of view, we should be able to impose it. 
**Paul Christiano** *[21:21]*: I think a way we could do a FLOP constraint if we wanted to do it would be provide another resource. So you still have a time constraint that you just have some. You're given some box that just runs code. You're allowed to use lots of processors as long as it's. You're just doing tensor contractions and so you can use X flops of tensor contractions for free. I don't know if we could set up something like that. 
**Mohanty** *[21:40]*: Yeah, we should be actually. 
**Jacob Hilton** *[21:44]*: I mean, is flops even the right thing for. 
**Mohanty** *[21:49]*: Yeah, exactly. I'm not really sure. 
**Paul Christiano** *[21:50]*: I don't know. 
**Mohanty** *[21:51]*: I was first quickly thinking about the technical feasibility of it and then second is. Yeah, of course from the problem design you would have to kind of think through. 
**Paul Christiano** *[22:02]*: Yeah, we can look at the structure of our circuits are just more annoying because everything is less uniform. Like they're not big dense mat moles. It's some weird sparse operation. 
**Mohanty** *[22:12]*: But I think, you know, in the first iteration of this kind of adding all these complicated interfaces for them to kind of get used to would kind of add to the barrier to entry. Right. So I would keep the resource constraint as simple as we can and just although ensure that the compute hardware they're running is something, you know, that they're very sure that it has expected performance that they can also debug themselves. 
**Paul Christiano** *[22:38]*: Yeah. From my perspective, the nice thing would be if we could provide methods that are performant for circuits. Like if we provide a forward pass method, if we provide just a really performant implementation built on reasonable building blocks for a forward pass which is basically the same as mean prop and then more interestingly for covariance propagation. I just don't know if we're actually good enough at performance engineering to get an optimal implementation for those things. 
**Mohanty** *[23:00]*: No, just to kind of address that whole aspect that you were trying to focus on with the perform with the point of performance engineering, all the baselines do not have to pass through the leaderboard. In fact, my point with baselines is one is we basically see a clear point, a clear anchor on the leaderboard. Second is baselines themselves is like more they have. They more have an educational purpose to them. So basically your same baseline I come up with the simplest implementation of it. Then I say okay, repo one basically the simplest implementation still uses the same implementation, but it won't pass. Now let's go into the second version of it, which basically does it a little bit more efficiently. That's when we jump in with a few of the performance optimizations. 
**Mohanty** *[23:43]*: Then we said, oh, it's too complicated, but let's just do it with C bindings and which is also. They can also do as a part of the submission and then we help them understand. We say, oh, but across all of them, only these two pass. The first one does not pass through because of performance reasons. And that's perfectly fine. 
**Paul Christiano** *[23:57]*: Yeah, I guess I'm mostly thinking about this from an accessibility perspective. So we're comparing performance. The performance of sampling of running a bunch of inputs through the model. The computational cost of running a bunch of inputs through the model may depend a lot on how you implement the forward pass, like your batch forward pass on the circuit. And so we really want to have a batched forward pass implementation that's really good so that people don't. Because if we don't do that, someone could just spend a lot of their time being like I improve the performance of sampling by like 30% or something. And that might be more important than the algorithmic improvements. 
**Paul Christiano** *[24:32]*: So it feels great from an accessibility perspective, if we can supply some building block, like if we had to do anything in C, we can just supply the Python implementation at the end that calls out to that and then people can use it as a black box rather than having to do any of it themselves. 
**Mohanty** *[24:45]*: Yeah, I think that's also easy. 
**Paul Christiano** *[24:46]*: Right. 
**Mohanty** *[24:46]*: We just have multiple implementations of the class. Right. One is like a hyperF1 and then the interfaces are the same. 
**Paul Christiano** *[24:57]*: Yeah, I have to think about it. Numpy may also be fine. It's possible his implementations are fine. I guess maybe my question was this seems like a thing we should deal with on our end if we are as part of deciding to what is the accessibility of circuits and do we want to do a test on circuits? Anyway, sorry for that digression a bit. 
**Mohanty** *[25:22]*: No, I think that's fine. That's also someone from our team can also kind of help in kind of figuring out because again the ensuring that the start ticket itself is as an entry point for the participants is very accessible is also part of like our mandate there. 
**Paul Christiano** *[25:38]*: And are there any questions about scoring? I think scoring is like a little bit fiddly and we could also provide that in some kind of repo or we could just talk with you and get the high level description. But it seems like scoring depends on how much time you take. We want to Evaluate algorithms at a bunch of different time budgets. We want to make it easy for people to know is the question about your environment. Can people tell really well how fast something will run on the contest environment? 
**Mohanty** *[26:02]*: Yeah, because we will just, yeah, we will run it on an AWS instance and they can actually spin up that instance and run it themselves. And also we are not giving them huge instances here. So again, it won't be also restrictively expensive. And in principle, again, it wouldn't be hard to kind of speak to someone at Amazon and say, you know what, we're running this. How about give some credits for the participants? They usually are very happy doing that. But in terms of scoring, yes, I think that basically would need a lot more diving deeper into where we need a very clear idea of what the scores are. And also all these adversarial mechanics which might happen when we start including this performance component in the score, time component in the score, etc. 
**Mohanty** *[26:46]*: Because then they start kind of exploring a part of the solution space that we don't really want them to focus on, but it's important for them to focus on to rise up the leaderboard. 
**Jacob Hilton** *[26:59]*: I think the rough proposal was. We impose some time constraint and then we allow, we give some sort of performance bonus to people that run within who run faster than that, but only up to say, 10% faster or something. 
**Mohanty** *[27:22]*: Yeah, I think there one thing could be you could come up with a performance threshold that can also be dynamic. We only decide what the threshold is at the end of the competition. Then, say, among these top submissions which have crossed this percentile of the score distribution, we give a separate prize to whoever is the fastest. 
**Paul Christiano** *[27:44]*: That way, the way performance works on this task is probably a little bit unlike most tasks, roughly speaking, you can spend 1% more compute to decrease your error by 1%. Like, there's a pretty linear relationship between compute and error. And so when we talk about a bonus, we just mean translating directly into, like, if you run 5% faster, we just decrease your error estimate by 5%. Mostly people, like, all the solutions will probably be like roughly the same kind of curve. Like if we draw a graph of how much time they take versus how much error, they'll all be about 1 over n, but just they'll be shifted down a little bit and maybe they'll have a slightly better slope, but mostly they'll have that slope, I think. 
**Mohanty** *[28:20]*: But then I think we can empirically validate this already be running some experiments at our end, assuming again we can. And then I think, yeah, have some notion of adjustment at the end for this performance track of sorts. But these could be also separate tracks altogether. We don't need to give all the prizes based on the same leaderboard where there's a separate leaderboard that people who only care about the performance track come in and take part. 
**Paul Christiano** *[28:47]*: Yeah, I think the problem with this task is because it's always possible to spend. Spend more time to get better results. I think it's really. It's very impossible to have anything that's not. We can't have separate tracks. There just is the notion of performance because someone could always. You can always just spend twice as much time and get a better estimate. So competing on quality is the same as competing on speed, but it's still bounded. 
**Mohanty** *[29:08]*: Right. In context of performance, there's just maximum amount of compute and time that you can use already. 
**Paul Christiano** *[29:15]*: I mean, we would impose that. Yeah, I mean, our ideal would be people just use exactly like it's a hard threshold and everyone uses exactly the same amount of time. The only reason to have the flex is because maybe it's a little bit fiddly to write your algorithm to get it exact, or maybe there's a little bit of variability. Maybe. Another question I had from an accessibility perspective is it seems like people spinning up AWS instances is fine. It seems like the task is easy enough that people might just run locally and do dev locally. Maybe that's not right. But especially in smaller circuits. 
**Paul Christiano** *[29:44]*: I don't know if it might be nice from an accessibility perspective if it's possible to offer the service of just people submit their code and then they get immediate feedback one circuit of did your thing run and how long did it take on that circuit? And then maybe they never even have to go through AWS machine. They never have to set up the contest environment themselves. 
**Mohanty** *[30:02]*: Yeah, we can do that. Yeah, that shouldn't be. But usually when someone is doing some kind of optimization, they would want full access to the node so they can profile it. Right. Because we won't know what kind of profiling they would want to do. 
**Paul Christiano** *[30:15]*: Yeah, people who are doing a good job probably will spin up instances, but it's. 
**Mohanty** *[30:18]*: Yeah, for others, we can definitely provide them that feedback. Actually, we will anyway provide them that feedback for every submission they make. So for every submission they make, we anyway want to kind of provide them a rich, detailed dashboard. And also you can actually check it in the chess challenge where just because were using these Trainium instances that many people are not used to, so we had to kind of give them a lot More feedback. So they have a bird's eye view of what's happening. Because many people were training on NVDA hardware and then getting evaluated on trainium ones. So we just want to be safe. 
**Jacob Hilton** *[30:49]*: And do we want to discuss this idea of having the same algorithm should work for quite different compute budgets. So the algorithm is given at runtime like a circuit and you just get the circuit just given a number which is how long it has to run and. And then it has to be adaptive. Do we want to try doing something like that? 
**Mohanty** *[31:12]*: I would love to try something like that. Because again, the distribution of the models we get around it are kind of doing some analysis of them would be super cool. My concern was this whole if else regime thingy that happens, which was not particularly useful the last time we did. But here you say, okay, that does not matter, then I'd be curious to basically go ahead and specifically try. But again, if we can already define the regimes as a part of the problem difficulty, that would already be quite easy. Then again, it's not a continuous regime scale and everyone is kind of picking up different regimes with their own if its conditions. But if we can also already kind of classify, it becomes a lot more easier to have some coherent understanding of some behavior out of the whole collection of models. 
**Mohanty** *[31:52]*: Because again, it's exactly experiments like these where what you get out of the challenge is not just the leaderboard, but all the submissions. Right. So. And not a lot of these challenges actually go and look at the distribution of the submissions to try and see what's kind of happening. But then recently we ran this chess one. Suddenly I have like 4000 or so of these SLMs, like 7 to 8B param models, all of which they are trained on chess. And I already know how they're performing against stockfish anchors. So then what do you do with them? Right. So that's another set of things that I really want to kind of unlock through these experiments. So it'll be good. 
**Paul Christiano** *[32:27]*: We often make graphs for ourselves of this, like performance versus or like time versus error. It'll be cool to be able to see like if each participant gets evaluated a bunch of different points. It would be cool to just like see that curve. 
**Mohanty** *[32:38]*: Yeah. 
**Paul Christiano** *[32:39]*: In fact, right now like. 
**Mohanty** *[32:42]*: I can. 
**Jacob Hilton** *[32:43]*: Actually weighting, I guess, I mean we could probably just like equally weight it in log space or something. 
**Mohanty** *[32:49]*: I don't know. 
**Paul Christiano** *[32:50]*: That was my. My suggestion was like weight each of them by like the error of sampling on that data point and then just be like, what's the sum of the rate your ratio over sampling. 
**Jacob Hilton** *[32:58]*: That's a good idea. 
**Paul Christiano** *[32:59]*: Yeah. 
**Mohanty** *[33:00]*: I'm just dumping down a bunch of plots from the CHESS one that were recently generated just to get a sense of the stuff you can in principle. Oh, okay. They do not make it easy to share anyway. I will just share it over a chat or something. But I was trying to find how. 
**Paul Christiano** *[33:22]*: To attach something here only available to invited users who are in the host's organization. 
**Mohanty** *[33:33]*: But what I will do is I will just create a public drive and share it with you. 
**Paul Christiano** *[33:40]*: In terms of what our next steps, it seems like there's us preparing better documented evaluation and starter or like evaluation code and baselines. I don't know what the step is on the performance engineering aspect. I guess we talked about this a bit, but is that something that we should think of as in our court, we should try and get the very fast implementation good, or is that something we should work with you on? 
**Mohanty** *[34:12]*: It would be both. Right. But then first you have to kind of, if you already have one implementation at your end, that will be helpful because again, for us, again it will just take a disproportionate amount of time. But then the goal was to figure out how the structure of the starter kit evolves around the existence of a much faster performing class for the same one. In terms of task, I think right now what we need is to get things moving is like a single document where we have actually technically described the benchmark using every single thing that are needed and that would need a lot of back and forth. 
**Mohanty** *[34:44]*: For example, when we say what tracks we are running or what the scoring function is, we actually need to have this one document as a single source of truth because there will be multiple people from our team who would be working on this. And then I would want anyone from your team and our team to just use this document as a reference. Then we would just create a single repo where we would have these working versions of actual implementations as well, which would be similar to the repo that we have, but along with some other performance profiling scripts, etc. Included our end and all of them also reference from that document. Then the first step was literally the same the repo that you have mentioned. But here the next goal would be to document everything in that single source of truth. 
**Mohanty** *[35:32]*: Does that make sense? 
**Paul Christiano** *[35:36]*: Yeah, that makes sense. It sounds like the. Yeah. So next step is writing some of that code, documenting it well, and preparing a first version of this document that just describes everything in full detail. And then we start a collaboration from There. 
**Mohanty** *[35:50]*: Yes, exactly. So the goal would be this is like a, a technical report that we hand to anyone who's just interested in just getting curious and fiddling around on the problem. Let's not worry about them building a whole model altogether and how we can measure the scores from the benchmark. And once we have that, we will start working on figuring out how to do this well and at scale there again, if anything new comes up along the way, we will just make those decisions on the issue tracker there itself. So again, these decisions are not just trapped in our email thread. And finally, once this is mature enough, where we say, you know what, let's freeze a version of it, we will translate that into our internal evaluation framework and start putting together the draft websites, etc. 
**Mohanty** *[36:37]*: So that you can already start making changes to it. 
**Paul Christiano** *[36:43]*: I guess one thing we'd imagine doing was doing a test solve with someone who is adjacent hasn't worked on this problem in the past. What stage in the process would that go? Should we do that like before we're finalizing and sending over to you, or should we do that like on your infrastructure, like with preliminary version? 
**Mohanty** *[36:59]*: Yeah, I think the experience should be end to end so that again, we always know how people kind of feel when they are making submissions here. So any tools, etc. That we develop, they also have to kind of test that. So the idea would be we have this first very drafty version of this repo, where we have one version that we are already confident about. For example, the one that you have mentioned, we just develop everything around it and then we will release multiple versions of it. So anyone who is testing it, they should be able to make a submission to our infra, get a feedback and then see their name on the leaderboard and let it rise up. Then they basically come in and also help us build these diagnostic dashboards, etc. 
**Paul Christiano** *[37:38]*: So the order will be we get a basic version in place, we submit it, or we send it to you and we collaborate to make it into a version that is a bit more polished and runs on your infrared. Then we'll do some test solving against like a beta version that is actually running on your infra so people can have the whole experience of seeing the leaderboard and issues with feedback and everything. 
**Mohanty** *[37:57]*: Yeah, exactly. And then there also if you want to kind of. And it would be invite only, right? So that the rest of the aircraft community cannot actually see it. So anyone who you give us a list of emails of, they will only be able to see it and then they would Pretty much have this private invite only competition just for them and we can do multiple rounds of this, right? So if you imagine if you run two or three in person events, each of them with just five, 10 people as well, then we can have multiple different leaderboards of each of them and if we want we can make the solutions of the first round public for the second round to actually also benefit from the exploration that these participants have done. 
**Mohanty** *[38:32]*: And then you can separately figure out all the incentives etc that would be needed. 
**Paul Christiano** *[38:41]*: I guess it sounds like this will be the part I am most afraid of remains like writing starter code that does the optimization in C and then exposing it in an abstraction that competitors can use so they don't have to write a bunch of C themselves. That's the part I'm most scared of. Other than that everything seems straightforward. 
**Mohanty** *[38:58]*: I think that in the worst case we can do. In the worst case I think we can basically go ahead and do that. Usually the way I do that is I try to kind of hire a research fellow who is a past AI crowd winner in some of the Neurips competitions. Then I say come on board and here we have this thing and I knew that this person has won because he went all in on optimizing then basically trust that within a few weeks he's going to come up with something really amazing. But that is something I feel confident that we should be able to do in a way where the interfaces are nice, static, it isn't bloated and participants. It's a treat for the participants to use it. 
**Paul Christiano** *[39:35]*: Yes, I guess we will do a first version in the rough unpolished version. We'll not do any of that optimization. It will just be reasonable numpy and then that will get baked on your side. We can do test solves in parallel with all that. We can find someone, either a test solver or one of us or someone else who can do a really intense performance pass and figure out how to structure the performance starter code. 
**Mohanty** *[39:56]*: By the way, if we already have the basic NUMPY version that we know analytically works well, can't we just trust an agent harness to go all in. 
**Paul Christiano** *[40:05]*: If they're able to do it? Yeah, I would try it first. Yeah, first ask Codex what's your best implementation of this algorithm? 
**Mohanty** *[40:16]*: No, no, basically the hardness would be okay, here's the implementation. First we test check it against the Python, our NUMPY implementation and then measure it based on performance and leave it running for CLAUDE sessions as well. 
**Jacob Hilton** *[40:28]*: Yeah, I wouldn't mind getting My hands dirty with some rust again. I haven't done that in a while. 
**Paul Christiano** *[40:39]*: Yeah, it seems like everything is pretty. This seems like it should be pretty good contest and should not be that much effort. I guess we'll see what happens. So it's possible maybe one other thing to flag. When we do initial test solves, it's possible we'll be like, actually circuits are kind of a pain in the ass and like a really unpleasant setting and open. We want to be to changing domain at that point. Probably we want to basically be locked in. 
**Mohanty** *[41:03]*: Yeah. I think that really depends on how carefully we communicate. Both in like all the communication we do around the challenge, but also the legal rules that participants accept. Right. So that if we just do carefully, then I think it should be fine. But we have done this in the. 
**Paul Christiano** *[41:17]*: Past, especially for problems during test solving, not after the contest becomes public. Like most likely during test solving, we would just change the depth of the circuits or change some parameters and it would still be all the same code. 
**Mohanty** *[41:29]*: How big a crowd are we expecting in that whole beta test solving phase? 
**Paul Christiano** *[41:35]*: I mean, if it was just us, we would probably bring one or two people. I'm imagining one ARC employee test solves and one other constellation person test solves or something. I don't know what you were imagining. 
**Jacob Hilton** *[41:46]*: Jacob, but I was maybe imagining more like five to ten people or something. But I don't know. 
**Paul Christiano** *[41:54]*: It just depends how long because we have a lot of people who test solve for a few hours and then a few people who test solve for a day or two. 
**Mohanty** *[42:01]*: And then we could have a few variations of these. Because the changes of the variations you have in mind, they're not wildly different from an evaluation implementation point of view. So we just have the evaluators ready and then let people go fiddle around and come give us feedback. But then they have to dive deep and go all in on at least one. So we learn some more long horizon feedback as well. 
**Paul Christiano** *[42:27]*: Yeah. My biggest single concern is that the high depth circuits may just be like too hard. I guess that's as you say, you can have a blue sky challenge where like beating sampling is really hard once you get up to these large depths. But we think it should be possible. But it's maybe very hard. Like it's fine. Our proposed scoring procedure was like you get scored over a lot of depths simultaneously. 
**Mohanty** *[42:45]*: Exactly. 
**Paul Christiano** *[42:47]*: And so it would be fine. People would just get most of their point savings in their earlier depths. But they may want to change the structure of what we actually are evaluating. 
**Mohanty** *[42:57]*: As long as we have some methodology in which we know the distribution of submissions won't basically lead to a saturation of the benchmark, at least during the competition. Then then we are good. Right? Because else we are in a point. At the end of the day, we'll. 
**Paul Christiano** *[43:09]*: Be thrilled if it got saturated. If it gets saturated, we'll be thrilled with the outcome. If this benchmark, that would be an extremely good, extremely successful outing. We found God's algorithm for this problem or like the right algorithm, optimal here. 
**Mohanty** *[43:24]*: I was mostly thinking from the legal implication of it, but then yeah, I'm sure there's solutions around it. It's mostly. Then we are in a tricky position where we have promised so many people prizes, but now we have no statistical way of kind of defining who actually gets the prizes. Then in those cases it gets into really tricky territory because this is called as these competitions and the formats legally are called as skill based promotion. So we have to really demonstrate how whoever wins any cash prize, you can't. 
**Paul Christiano** *[43:50]*: Randomize. 
**Mohanty** *[43:53]*: Who gets the cash prize based on skill. Except in many. Without that in many countries a lot of these anti gambling laws come in place. So you have to be really careful. 
**Paul Christiano** *[44:03]*: But then we can add our settings. Like we can keep generating more statistical power indefinitely by spending compute. So if there's like we could just investigate harder. It seems like it would also be totally fine if several people are tied at the top distributed across the tide. Like not to randomize, but distribute it evenly. If we can't significantly distinguish two leaders, we can just give it to everyone who's statistically indistinguishable from. 
**Mohanty** *[44:29]*: I think we have to be careful about that in the rules, especially in problems which are fairly new. Right? 
**Jacob Hilton** *[44:36]*: Yeah. 
**Paul Christiano** *[44:37]*: There will probably be a lot of dispersion though at the top end would be my guess. It'll probably be easy to distinguish them unless people do surprisingly well. 
**Mohanty** *[44:47]*: I think here the whole goal would be in a challenge like this, what happens is you launch the challenge the first two, three, four weeks. Again, the participants are confused. They're fiddling around, maybe trying the baseline and whatnot. So in that duration it'd be good to basically put in a little bit of, you know, dev rel effort or more like, you know, educational effort to kind of really get people on board it. So there I would be curious and trying and seeing if there would be, you know, some effort or energy from your side in kind of also helping make that happen. Because again, without that we are not really sure, you know, how the trajectory of a challenge like this On a completely new problem would go because we will be doing. 
**Paul Christiano** *[45:26]*: It will be related or running in parallel with us doing a ton of outreach about what we mean and what the problem is. So that will probably spill over some. I think it is an important question for us to think about like how much staff time should we be setting aside to do various types of contests and that will also affect like how we think about which contest we actually want to do. I think it is a somewhat different world for us if we're like we should allocate, you know, two weeks of staff time during the period of the contest to like help participants and like drum up support versus if we should allocate two months of staff time. Like those are pretty different scenarios given our level of staffing. I think we are buckled in for two weeks at least. 
**Paul Christiano** *[46:07]*: I think like during the contest of, you know, a person is half time or third time or whatever looking into it, but are probably not expecting to have someone like full time during the course of the contest or to have a full time equivalent during the course of the contest or engaging with people. 
**Mohanty** *[46:22]*: Yeah, no, that's usually the expectations of whenever we can have external interface where the idea is most of the support for participants and whatnot is already done by someone from our support team. And a few questions again which are extremely technical related to benchmark design or some baseline stuff they might bubble up. But the effort that would be needed in terms of view for support for participants should be extremely minimal. For you the time that would be needed is we usually organize these town halls where that would be at end of the first round, beginning of the next one where we probably. It would be good if you can come in and give a talk to the participants actually engage in that one hour session. 
**Mohanty** *[47:01]*: Second is any resource that we create in the beginning there it would be good to really put in some thought on how to communicate this well. And there also I don't think it should be like a multi week effort. But the few serious days of focused work already does wonders there. Right. Because again we will have someone from our end really focused on getting that to work. 
**Paul Christiano** *[47:23]*: Yeah, I mean we will be. I think in total we will be spending a significant amount of time defining stuff and helping with test solves and communicate and so on. 
**Mohanty** *[47:30]*: Yeah, A majority of it would be in the benchmark design actually at your end. 
**Paul Christiano** *[47:34]*: Yeah. Maybe one question we haven't talked about is like the size of prizes and like where you're at in terms of. I don't know, maybe even Jacob discussed This and I missed it just in the email thread, but I assume we would both be providing prizes. We should talk about what size of prize makes sense and then also paying you guys for support. And I'm just interested in having at some stage, like maybe soon we should talk about rough. Get a sense of rough numbers for those. 
**Mohanty** *[47:58]*: Yeah, again we don't have like the public numbers around these because it very much varies around the effort around challenges like these. But again this is like a code submission challenge which would be standard across many of the Neurips challenges we run or like the corporate ones. So we can get to the prizes per se. That's separate because you can give the prizes either directly to the participants or via us. When you give it via us. Again, we will have to basically take a small administrative fee to cover for all the headaches that come along the way. In context of how we cause challenges like these, they end up being something of the order of about 85 to 90,000 Swiss francs. 
**Mohanty** *[48:31]*: But then that's for the whole six to eight month period engagement that would be needed and that covers pretty much everything along the way including support for participants, implementation of starter kits, baselines and maintaining the whole challenge throughout the whole duration and all communications, design, etc. That are needed. 
**Paul Christiano** *[48:50]*: Cool. I think that was in the general ballpark were imagining. So that sounds great. What size prizes are typical for Neurips competitions? 
**Mohanty** *[48:58]*: Yeah, so Neurips competitions they vary a lot. Usually they are. Some of them they are in the range of 10 15k ish, but we have run many around the 30 to 50k ish ranges. I think recent years had higher as well. One rule of thumb we try to use is anything beyond 50k. It slightly has diminishing returns in terms of how people perceive it. So one thing we have done is beyond any 50k prices we would add a very small budget and give you give some kind of experiential prizes or something tangential like a little army of drones or PlayStations we ship to them or travel grants etc. Those have worked out a lot better than increasing the prize pool from 50,000 to 60,000 per se. But then of course increasing it from 50,000 to 100,000 basically does add the perceived value. 
**Mohanty** *[49:52]*: But there at least at this stage the question should be again, would it make sense to kind of have some well thought out blue sky prices which are a little bit higher where again you also try to go ahead and say we don't announce it at the beginning but we try to ask around anyone Other organizations as well who are interested in pitching into this blue sky price pool, if this particular threshold is met, etc. But my recommendation would be to kind of stick around something in the range of this 50, 60,000 ish and leave aside a set of the price pool. 
**Paul Christiano** *[50:24]*: So. 
**Mohanty** *[50:25]*: So you can give out some incremental prizes based on things you learn as a part of the competition. 
**Paul Christiano** *[50:33]*: Yeah. Is it common to have like we discussed this prize for like best algorithm or like best contribution, where like we then look at descriptions of the algorithm and are like, you're only eligible for the prize if you have a clear description of what you did and it constitutes meaningful algorithmic progress. Is it typical to have additional prizes with that kind of structure or do you think that's something that makes sense? 
**Mohanty** *[50:53]*: Yeah, there the question is, do we have the bandwidth to actually review them and give them feedback within a fixed amount of time? Right. So we have given a lot of community contribution prizes, which actually would make a lot of sense. Exactly. In a challenge like this where we say for round one we are going to give XYZ amount of dollars for community contribution prizes. Again, there can be multiple versions of these, but this is mostly for people who create and share resources which basically help other participants. Or these could actually be videos people basically did, explaining, you know, a baseline or explaining the problem. Or these could be notebooks that people share with some solutions, but really well documented. Or these could also be just archive write ups of their solution, etc. 
**Mohanty** *[51:36]*: But there the bottleneck is can we actually review them well and give them feedback in a reasonable amount of time. Right. Which also other participants agree with. 
**Paul Christiano** *[51:45]*: I think the way we most imagine this is if you take the top couple submissions, we would only want to give prizes to algorithms that are quite good. So if you're just looking amongst the top few submissions, looking at them, maybe the top three or four submissions just being like, do any of them have a clear explanation of what they did and then deciding amongst them rather than evaluating. We wouldn't want to read submissions that are not for very good algorithms. 
**Mohanty** *[52:09]*: If it's only for prize recipients, it's a lot more easy because then what we can do is in the rules we say the prices are conditional on you releasing your code under an, you know, open source foundation approved license of your choice. So they have a little bit of flexibility and you giving a two to four page write up using this template. And this is something we have done. In fact, this is other interesting aspects because in many cases, especially where we ourselves have trained the baselines. What we do is we also care about how the solution space looks like. So at the end of the competition we give the prizes. We say oh by the way, you were competing until now all of you are teams, but some of them we also hire as research fellows of sorts. 
**Mohanty** *[52:49]*: Then we spend a two to three month period, ish where everyone collectively documents the solution space they have explored. Redo some experiments which come up from the shared understanding of what other people have done. Then gather the results into like a broader solutions paper of sorts. But that is a very intensive exercise so we don't necessarily have to do it but just kind of helping you understand some ways in which this could take off. 
**Paul Christiano** *[53:17]*: Do you have other big questions on your mind, Jacob? Are there other things we need to talk about? 
**Jacob Hilton** *[53:21]*: That was the main things I think probably it's good to just to figure out like next steps on each of our sides, like when we should check in next and what we should try and like have in place by then. 
**Mohanty** *[53:34]*: Yes. So one thing I would do is I would definitely over today, tomorrow I would put together the sow. So you know, you have a much more detailed understanding of, you know, what are the services provide all that stuff. That's mostly on the admin side but it's still good to kind of have a clear view for you then I think from our end I would definitely be interested because again it's really well aligned with this thing. I wanted to do like a series of challenges around alignment etc to really mobilize the community around this and I will figure out who else from our team would jump in on this. But I think at least at this stage we need at least probably just one or two technical and research people until we get to that point where we have this repo. 
**Mohanty** *[54:12]*: We say now let's translate it into our evaluator there. I would be willing, I would set aside some time when we can actually more regularly meet in sync and if we have a good way in which we can sync a sync like Slack or any other place that would actually make it a lot more easier. 
**Jacob Hilton** *[54:28]*: Yeah, we could probably do Slack Connect. We have a Slack instance. So I guess presumably then I can. 
**Mohanty** *[54:37]*: Just create a Slack Connect channel and then send it across. Send across invites to both of you and you can add others from your team, whoever need to be there. 
**Jacob Hilton** *[54:47]*: Great. 
**Paul Christiano** *[54:48]*: We should probably have an ARC Slack someday anyway. Different discussion. Yeah. 
**Jacob Hilton** *[54:58]*: Okay, so we should basically start preparing. Documentation and write ups of some kind of write up of what the contest should look like and criteria yeah, exactly. 
**Mohanty** *[55:16]*: I wouldn't impose any structure at all. Later we will reorganize it into a structure and help you understand how interfacing with our stuff becomes easier. First, right now, we usually like that report. You have full reins on how you want to organize it. Just dump in as much content as we can. I'll mostly care about the code interfaces where we can just send you a pull request on the repo already so at least we can start running some experiments in our end. 
**Jacob Hilton** *[55:42]*: Sounds good. 
**Paul Christiano** *[55:43]*: Yes. We want to have a repo that's in good shape with baselines and evaluation code. And then we want to have documentation for the full scoring rules and scripts that actually run against that repo and stuff like that. 
**Mohanty** *[55:54]*: Yeah, exactly. Right. So whenever we are talking about a specific problem formulation, we not only need the description of it, but we also need to agree a high level code example of how we are evaluating so there's no chance of some information being lost in the translation. And as we create multiple versions of different problems, we can basically figure out how to organize them later. But how the score is calculated. If we have that as a clear code example, even if at a high level, that really makes it a lot more easier to not make mistakes further down the pipeline. 
**Paul Christiano** *[56:27]*: Yeah, I guess we could even write timing code. It'll be a crappy timing, but we can just. For the purpose of communicating what's happening, we could write an evaluation script that runs at different time budgets, feeds the time budget into the submission, blah, blah. 
**Jacob Hilton** *[56:41]*: It seems like we basically just want to self contain thing that like would be enough for someone to basically like run and their own version of the contest hopefully like. 
**Mohanty** *[56:51]*: Yeah, exactly. Right. Because they would be running the whole local evaluation and all the assumptions we try to communicate in the overview, they should be included in that report, right? 
**Paul Christiano** *[57:01]*: Yeah. Okay. 
**Jacob Hilton** *[57:06]*: Yeah, we should think about who wants to do that. 
**Paul Christiano** *[57:09]*: Yeah, we should make plans internally. 
**Jacob Hilton** *[57:11]*: Yeah. 
**Paul Christiano** *[57:12]*: And then we'll also sync up on the statement of work. What is the timeline for next meeting? On these two fronts I would be. 
**Mohanty** *[57:18]*: Actually statement of work. I can send you across today, evening or tomorrow latest. Then for the meeting, can you suggest something? Because again my schedule is going to be a little bit all over the place because I'm india actually right now for the next two weeks. But again I'm kind of working also across time zones so would be good to know what slots are comfortable for you. Then I can just figure it out around mine. For example, this slot is perfect, like Monday evening here and probably like early in the day for you can work perfectly for me. 
**Paul Christiano** *[57:54]*: Excuse me. 
**Jacob Hilton** *[57:56]*: I think. Yeah. From next week on, Mondays are going to be a little busy. I mean we could just use the slope. Yeah, we could just use. 
**Paul Christiano** *[58:08]*: This is before at least. We can also maybe go a bit earlier if we want to have time. Earlier. I don't know if you're having. Earlier is good for both of you. 
**Mohanty** *[58:14]*: I think earlier would all. You mean earlier in the day, right? 
**Paul Christiano** *[58:18]*: Yeah, like an hour earlier or something like that. 
**Mohanty** *[58:21]*: It works for me. 
**Paul Christiano** *[58:22]*: It's great. 
**Jacob Hilton** *[58:23]*: I probably prefer this slot, this lot. 
**Paul Christiano** *[58:25]*: Okay. The main thing was do we want to have a space for. Yeah, we can just start meetings at 10. 
**Jacob Hilton** *[58:30]*: Yeah, yeah, I think it's too. 
**Paul Christiano** *[58:31]*: People might be okay. 
**Jacob Hilton** *[58:34]*: So should we plan one week from now or two weeks from now or. 
**Paul Christiano** *[58:37]*: What do we want to do? 
**Mohanty** *[58:37]*: I think one week for sure. And then I think we will probably come up with a few more task items while we discuss on Slack. So of course any updates etc. To the admin on the statement of work admin stuff be good if we can just coordinate on email because then I can loop in the legal team there and on Slack any other technical stuff that come up. We can basically figure out some clear milestones so that again when we meet next Monday again we have a more tangible sense of the problem or clear idea of what are the next steps that need to be done. 
**Jacob Hilton** *[59:07]*: Great. Okay. 
**Paul Christiano** *[59:09]*: Okay, cool. 
**Mohanty** *[59:12]*: All right. Cheers. Have a good bye. 
**Jacob Hilton** *[59:16]*: See ya. 
