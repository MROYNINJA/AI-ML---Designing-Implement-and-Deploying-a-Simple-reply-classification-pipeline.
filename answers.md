# answers.md

**If you only had 200 labeled replies, how would you improve the model without collecting thousands more?**  
With only 200 examples I would use data augmentation (back-translation and paraphrasing) and transfer learning (fine-tune a small pre-trained transformer) to leverage large-scale language knowledge. I would also use cross-validation and few-shot techniques, plus active learning to query the most informative unlabeled examples for labeling.

**How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?**  
I would audit model predictions across demographic and domain slices, add detection for unsafe phrases, and include a human-in-the-loop for flagged or low-confidence predictions. Regular monitoring, a bias-reporting dashboard, and conservative thresholds for automated actions reduce risk.

**Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?**  
I would provide structured context in the prompt (recipient role, company fact, recent event, desired tone) and include explicit constraints and examples of good/bad outputs (few-shot). Finally, I’d post-process outputs to check for factuality, hallucination, and to ensure unique phrasing per recipient.
