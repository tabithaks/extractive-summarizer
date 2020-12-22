# UN_Speech_Summarizer
Topic based extractive summarizer built to summarize UN General Assembly Statements (English).

### Sitemap
- `summarize.py`: Contains code for summarization, including LdaSummarizer class
- `Time_and_Sentence_Length_Analysis.ipynb`: Contains analysis of time it takes to summarize and sentence length penalty (penalizing longer sentences, making them less likely to be inlcuded in the summary).
- `un_debates_summarize.py`: Script to run summarization on all UN General Assembly Statements, saves summaries in summary column and writes to csv file
- `un-general-debates-summaries.csv`: Csv file that includes summaries in column headed "summary"
- `citations.pdf`: pdf file containing reference paper citations

### Answers to Questions (these are also at the top of the notebook Time_and_Sentence_Length_Analysis.ipynb)

a) Approach to the problem
Summaries describe the main ideas in a text, and I approached the problem with the objective of finding sentences that best cover the main topics discussed in each speech. With that in mind, for each speech, topic analysis (latent dirchlet allocation) was used to determine what the main topics were. The most discussed topics (based on the number of sentences a topic came up in) were then selected, and the sentence with the highest probability for each of these topics found. These sentence were extracted, ordered based on their when they came up in the orginal text, and put together as a summary. 

b) Measuring performance.
Below are possible ways to evaluate performance in this problem. Without extensive resources, I would start with the qualitative approach (perhaps developing a rubric for consistency) on a very small number of randomly selected text. Then I would then use the transfer approach in order to have a metric to compare this summarization method to others.

- Qualitative approach: an individual comparing output to actual text on a small subset of the data
- Resource intensive: if resources are available have people create ground truth summaries and combine the BLEU and ROUGE metrics to create an F1 metric representative of the lexicon overlap between automatically generated summaries and the ground truth created by people
- Transfer approach: evalute the summarization method on a different dataset of texts that already have ground truth summaries using the BLEU and ROUGE metrics. Use this evaluation as a proxy for performance on the UN General Assembly Statements summary problem.
