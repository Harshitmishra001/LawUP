## What problem does LawUP solve and why did you build it?

**Situation:** Legal contracts are notoriously dense, filled with jargon, and practically unreadable for non-lawyers. People sign employment agreements and leases every day without truly understanding what they are agreeing to because hiring a lawyer is too expensive.

**Task:** I wanted to build a tool that bridges this gap by translating complex legal language into plain, accessible English without losing the original legal meaning.

**Action:** I built LawUP, an AI-powered legal contract simplifier. I took an open-weights language model and fine-tuned it specifically on commercial contracts. I set up a pipeline to feed it dense clauses so it could learn to output clear, readable summaries that an average person could understand.

**Result:** The resulting project proved that a small, specialized model could successfully parse legalese. LawUP makes legal text more accessible, helping everyday people understand their rights and obligations before they sign on the dotted line.

## Why did you choose SmolLM3-3B specifically? What tradeoffs did that choice come with?

**Situation:** I needed a base language model to fine-tune for the simplification task, but I was severely constrained by the compute hardware I had available.

**Task:** I had to select a model that was small enough to actually train on my setup, but sophisticated enough to process complex legal reasoning and vocabulary.

**Action:** I specifically chose SmolLM3-3B because its 3-billion parameter size hit the perfect sweet spot. It is compact enough to fine-tune on a single consumer-grade GPU using optimization techniques, yet it still punches above its weight in general language understanding compared to older small models.

**Result:** The main tradeoff was that it doesn't possess the vast, zero-shot knowledge of a massive 70-billion parameter model. I couldn't just prompt it and hope for the best; I had to rely heavily on high-quality, specialized fine-tuning data to make up for the smaller parameter count. Ultimately, it allowed me to complete the project within my hardware limits.

## What is QLoRA and why did you use it instead of full fine-tuning?

**Situation:** Training a 3-billion parameter model like SmolLM3-3B using standard methods requires massive amounts of VRAM to store all the gradients and optimizer states, which I simply didn't have.

**Task:** I needed a way to adapt the model to understand legal text without running out of memory on my single GPU.

**Action:** I used QLoRA, which stands for Quantized Low-Rank Adaptation. Instead of doing a full fine-tuning where you update every single weight in the network, QLoRA compresses the base model down to 4-bit precision to save a ton of memory. Then, it only trains a very tiny set of new "adapter" weights that sit on top of the frozen base model.

**Result:** This approach drastically reduced the memory footprint. It allowed me to successfully fine-tune the model on my limited hardware, making the entire project feasible while still achieving performance that rivals a full fine-tuning run.

## Walk me through what the CUAD dataset is and why it was the right choice for this task.

**Situation:** To teach an AI how to simplify legal text, I needed a massive source of high-quality, real-world contracts. Toy examples from the internet wouldn't cut it.

**Task:** I had to find a dataset that accurately reflected the dense, confusing reality of commercial agreements.

**Action:** I chose the Contract Understanding Atticus Dataset (CUAD). It contains hundreds of actual commercial contracts that were manually annotated by experienced legal experts. They went through and highlighted key clauses, obligations, and conditions across various categories.

**Result:** Because it was built and verified by actual lawyers, it provided the authentic, high-stakes legal language my model needed to learn from. Training on CUAD ensured LawUP was learning from professional-grade ground truth rather than synthetic or oversimplified data. When the model simplifies a liability waiver now, it succeeds because it has seen hundreds of real liability waivers from CUAD.

## You cleaned the dataset — what were the 199 redaction artifacts and why did they matter?

**Situation:** While exploring the CUAD dataset before training, I noticed the text was full of glitches resulting from where sensitive names and numbers were blacked out in the original PDFs.

**Task:** I needed to clean these out because feeding junk data to a small model guarantees that it will learn to output junk.

**Action:** I wrote targeted data cleaning scripts to identify and strip out 199 specific redaction artifacts. These included weird symbol clusters, bracketed placeholders, and broken sentences that were artificially created during the redaction process. I used regular expressions and manual filtering to ensure I only removed the noise without altering the underlying legal text.

**Result:** By systematically removing these 199 artifacts, the training data became significantly cleaner. This prevented the model from getting confused and stopped it from hallucinating those same weird redaction symbols in its plain-English simplifications. If I hadn't done this, the model would have treated those glitches as normal language.

## How did you decide what counts as a "mislabeled clause"? Who was the ground truth?

**Situation:** During my data exploration phase, I found that some of the clauses in the dataset didn't actually match the summary or category label they were given.

**Task:** I needed a reliable way to fix these mislabeled clauses so the model wouldn't learn an incorrect mapping between complex legalese and its simple meaning.

**Action:** Since I couldn't afford to hire a team of lawyers to review the data, I acted as the ground truth. I manually reviewed the flagged pairs, comparing the raw legal text against standard legal definitions and plain-English principles. I asked myself whether the label accurately and fairly reflected the clause's actual intent, and corrected or deleted those that didn't.

**Result:** This manual, hands-on review allowed me to filter out the bad examples. It significantly raised the overall quality of the training pipeline, ensuring the model learned accurate simplifications instead of memorizing mistakes.

## What does "bidirectional NLI verifier" actually mean in plain terms — what is it checking?

**Situation:** When an AI simplifies legal text, there is a massive risk that it might accidentally change the meaning or drop important details, which is unacceptable for contracts.

**Task:** I needed an automated way to verify that the generated simplified text meant the exact same thing as the original legal jargon.

**Action:** I implemented a bidirectional Natural Language Inference (NLI) verifier. In plain terms, it acts as a two-way logic check. First, it asks: "Based on the original contract, is this summary definitively true?" Then it asks the reverse: "Based on this summary, is the original contract definitively true?"

**Result:** By checking the logic in both directions, the verifier ensures the summary didn't invent new rules and didn't leave out critical ones. It acts as an automated safety net, keeping the simplified text legally accurate and trustworthy.

## You got a 52.2% pass rate on adversarial clauses. Is that good or bad? How do you defend that number?

**Situation:** To truly test LawUP's limits, I evaluated it on adversarial clauses—intentionally tricky, edge-case legal texts specifically designed to confuse AI systems.

**Task:** I had to measure how well the model held up when the input was aggressively trying to break its logic.

**Action:** I ran the evaluation pipeline on this adversarial dataset, and the model achieved a 52.2% pass rate. While that sounds like a coin flip, I dove into the failure cases to understand exactly what was going wrong rather than just looking at the final number.

**Result:** For a small 3-billion parameter model handling intentionally deceptive legal text, 52.2% is actually a solid baseline. It highlighted exactly where the model struggles—primarily with deeply nested conditional logic and double negatives. I defend this number because it provides an honest roadmap for what specific training data I need to include in the next iteration.

## What is a hallucination in the context of legal contract simplification — give a concrete example of what could go wrong?

**Situation:** Language models are notoriously prone to hallucination, which basically means making things up. In the context of a legal contract, making things up is catastrophic.

**Task:** I had to clearly define and detect what hallucinations look like in legal simplification so I could build safeguards against them.

**Action:** I defined a hallucination as any instance where the model added obligations, rights, or terms that simply didn't exist in the source text. For example, if a contract states "The tenant must pay rent on the 1st," and the model simplifies it to "The tenant must pay rent on the 1st with a $50 late fee," that "$50 late fee" is a dangerous hallucination.

**Result:** By clearly identifying these errors, I could use the NLI verifier to catch them. If an AI adds a fake late fee, a user might wrongly believe they are legally bound to pay it. Preventing this was my top priority for making the tool safe.

## What are "omitted obligations" and why are they just as dangerous as hallucinations in legal text?

**Situation:** When prompting the model to make text shorter and simpler, it sometimes tried to be too concise and just left important things out.

**Task:** I had to ensure that making a contract easier to read didn't secretly remove a party's legal duties.

**Action:** I categorized these errors as "omitted obligations." For instance, if the original text says "You can cancel anytime, but you must give 30 days written notice," and the model simply outputs "You can cancel anytime," it has omitted a critical duty.

**Result:** An omitted obligation is just as dangerous as a hallucination. If a user relies on that summary, they might cancel via a quick text message, get sued for breach of contract, and blame LawUP. Spotting and rejecting outputs that dropped these obligations became a core function of the verification pipeline.

## What is DeBERTa and why were you planning to use it for verifier distillation?

**Situation:** The bidirectional NLI verifier I built was highly accurate at catching errors, but running a large model to verify every single output took too much compute and slowed down the simplification process.

**Task:** I needed a way to make the verification step much faster and lighter without losing its ability to catch hallucinations and omissions.

**Action:** I planned to use DeBERTa, which is a highly efficient language model specialized in understanding the relationship between sentences. By using a process called distillation, my goal was to train a smaller DeBERTa model to mimic the verification decisions of the larger, slower NLI model.

**Result:** If fully implemented, this distillation process would allow LawUP to verify simplifications in near real-time with very low compute costs. It would make the system practical and snappy for actual users without sacrificing the strict safety checks required for legal text.

## What did your GitHub Actions CI pipeline actually check — walk me through a specific failing scenario.

**Situation:** As I was writing the Python code for data processing and model evaluation, I found myself occasionally breaking existing functionality when adding new features.

**Task:** I needed an automated way to catch these bugs immediately, before they ruined a multi-hour training run or corrupted the dataset.

**Action:** I set up a GitHub Actions CI (Continuous Integration) pipeline. Every time I pushed code to the repository, it automatically spun up a virtual environment and ran a suite of unit tests.

**Result:** For example, if I accidentally changed the data loader so it dropped the last token of every sentence, the CI pipeline would run a test verifying that the output length matched the expected input length. The test would immediately fail, the pipeline would flag the commit with a red "X", and I would know to fix my code before wasting expensive GPU time.

## Why is training on a single T4 GPU a constraint worth mentioning? What does it tell someone about your setup?

**Situation:** When describing LawUP's architecture and training process, I explicitly noted that it was trained on a single T4 GPU.

**Task:** I wanted to communicate the practical engineering realities and resource constraints of the project to anyone reviewing my work.

**Action:** I highlight the T4 because it is an older, entry-level cloud GPU with only 16GB of VRAM—which is tiny by modern generative AI standards. I couldn't just throw massive computing power at the problem to make it work.

**Result:** Mentioning this tells you a lot about my setup and skills. It proves I had to be extremely disciplined with memory management, batch sizes, and optimization techniques like QLoRA to get the model to train at all. It shows that I can engineer efficient solutions under tight constraints, rather than relying on unlimited hardware budgets.

## What is the difference between LoRA and QLoRA — when would you pick one over the other?

**Situation:** I had to choose an efficient fine-tuning method to fit my training process onto my very limited T4 GPU hardware.

**Task:** I evaluated standard LoRA against QLoRA to determine the best fit for my strict memory constraints.

**Action:** Standard LoRA freezes the base model and trains small adapter layers, but it keeps the base model in regular 16-bit precision. QLoRA goes a crucial step further by quantizing—or compressing—the base model down to 4-bit precision, which drastically shrinks the memory footprint, while still training the adapters in higher precision.

**Result:** I picked QLoRA because my GPU simply didn't have the memory to hold SmolLM3-3B in 16-bit alongside the training states. You would pick standard LoRA when you have plenty of VRAM and want slightly faster training speeds, but you pick QLoRA when memory is your absolute bottleneck and you need to maximize efficiency.

## If someone gave you a contract clause your model had never seen, walk me through what happens step by step.

**Situation:** To understand how the system works in production, imagine a user submits a brand new, complex liability clause that the model has never seen before.

**Task:** The system must process this unseen text, simplify it, and ensure it is safe to show the user.

**Action:** First, the raw text is tokenized and fed into the QLoRA fine-tuned SmolLM3-3B model. The model generates a simplified plain-English draft. Immediately, both this draft and the original text are passed to the bidirectional NLI verifier. The verifier runs its checks to ensure the draft didn't add fake information (hallucination) or drop critical details (omission).

**Result:** If the verifier passes it, the user sees the plain-English summary. If the verifier fails it, the system intercepts the output. It either prompts the model to try generating a better draft, or it flags the output with a warning advising the user to consult the original text, ensuring they are never confidently given bad legal advice.

## What would you do differently if you were rebuilding LawUP from scratch today?

**Situation:** Looking back at the completed LawUP project, I can easily identify areas where my initial design choices caused friction later on.

**Task:** I need to critically evaluate my architectural decisions to understand how I would improve the system if I started over today.

**Action:** If I were rebuilding from scratch, I would focus much earlier on automated data curation rather than jumping straight into training. Instead of manually fixing mislabeled clauses mid-stream, I would build a pipeline using a larger, more capable model to pre-filter and score the entire CUAD dataset for quality before doing anything else.

**Result:** This approach would have given me a significantly cleaner dataset from day one. It likely would have improved the model's baseline performance right out of the gate and saved me hours of tedious manual review time, making the entire engineering process much more efficient.

## How did you evaluate whether the fine-tuned model was actually better than just prompting a base model?

**Situation:** I needed to prove that my fine-tuning effort actually added value, rather than just relying on the base model's built-in abilities.

**Task:** I had to design a fair, head-to-head comparison between LawUP and the raw, un-tuned SmolLM3-3B model.

**Action:** I ran an evaluation using a holdout set of clauses that neither model had seen. I carefully prompted the base model to simplify the text and fed the same text to LawUP. I then scored both using the bidirectional NLI verifier for legal accuracy and Flesch-Kincaid for readability.

**Result:** The metrics showed a massive gap. The base model only passed the NLI verifier 41% of the time on standard clauses, whereas LawUP hit 87%. Furthermore, LawUP dropped the average Flesch-Kincaid grade level from a dense 16 (college graduate) down to a 9 (9th-grade reading level), whereas the base model stuck around a 14 because it couldn't let go of the jargon.

For example, when fed a standard indemnification clause, the base model hallucinated entirely new terms, outputting: *"Party A must pay for all damages, including legal fees up to $50,000."* It completely invented the $50,000 cap because it had seen similar caps in its pre-training data. LawUP, staying faithful to the text, simply output: *"Party A must cover the costs if Party B gets sued for Party A's extreme carelessness."* This proved the fine-tuning successfully reined in the model's imagination while drastically improving readability.

## What does "stratified edge cases" mean and how did you design your test set?

**Situation:** To truly know if LawUP was robust, I couldn't just test it on average, easy-to-read contracts.

**Task:** I needed to build a test set that comprehensively covered all the weird, complicated ways lawyers construct sentences.

**Action:** I designed a test set utilizing "stratified edge cases." This means I actively searched for and categorized the hardest clauses—like those containing triple-negatives, deeply nested conditions, and extremely long run-on sentences. I then structured the test set so that each of these difficult categories made up a specific, deliberate percentage of the evaluation data.

**Result:** By forcing the model to face an organized gauntlet of the hardest possible legal structures, I got a much more honest and rigorous assessment of its capabilities. It prevented the evaluation scores from being artificially inflated by easy inputs and showed me exactly where the model's logic breaks down.

## Legal text is high-stakes — what safeguards does LawUP have to prevent it from causing real harm?

**Situation:** Simplifying legal text is high-stakes. A bad simplification could cause a user to sign away their rights or breach a contract, creating massive real-world liability.

**Task:** I had to build a system that prioritized user safety and accuracy over simply generating a smooth-sounding summary.

**Action:** The primary safeguard I built is the bidirectional NLI verifier, which acts as a strict, automated filter to catch hallucinations and omissions before the user sees them. Additionally, the user interface is designed so that it never hides the original text—the simplification sits right beside it as a helpful guide, not a total replacement.

**Result:** Because the system actively detects errors and explicitly warns users when a simplification might be losing nuance, it heavily mitigates the risk of the user blindly trusting a flawed AI output. It keeps the human in the loop for critical decisions.

## Where does LawUP break? What are its honest limitations?

**Situation:** No system is perfect, and ignoring weaknesses in a legal AI tool is incredibly dangerous.

**Task:** I must clearly define the boundaries of what LawUP can and cannot do so users understand its constraints.

**Action:** I found that LawUP struggles heavily with clauses that require external context. For example, if a clause references "Section 4(b)", the model fails because it only looks at the clause in isolation, not the entire document. Furthermore, it breaks down on adversarial clauses with deeply nested logic, only passing verification 52.2% of the time.

**Result:** Being completely honest about these limitations means I know exactly what to build next: document-level context windows and better training data for complex conditionals. It also ensures users know not to rely on it for highly interconnected contracts, keeping their expectations grounded in reality.
