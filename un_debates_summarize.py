import summarize
import pandas as pd


debates = pd.read_csv("../un-general-debates.csv")
debatestop100 = debates[:100].copy()
debatestop100["summary"] = debatestop100["text"].apply(lambda x: summarize.summarize_pd(x))

debatestop100.to_csv("un-general-debates-100-summaries.csv",index=False)



