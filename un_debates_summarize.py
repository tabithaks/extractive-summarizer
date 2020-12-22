import summarize
import pandas as pd


debates = pd.read_csv("../un-general-debates.csv")
debates["summary"] = debates["text"].apply(lambda x: summarize.summarize_pd(x))

with open ("un-general-debates-summaries.csv", 'w') as file:
    debates.to_csv(index=False)



