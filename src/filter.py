import pandas as pd
import data_helper
import matplotlib.pyplot as plt


# lower6 = pd.read_csv("../tmp/lt6.1_prob.csv",index_col='id')
up8 = pd.read_csv("../tmp/lt8_prob.csv",index_col='id')
up10 = pd.read_csv("../tmp/lt10_prob.csv",index_col='id')

predict = pd.read_csv("../src/submission_23_end.csv",header=None,names=['血糖'])

lowerfilter = up8.sort_values(by=['lt8_prob'],ascending=True)[:15]
lowerfilter10 = up10.sort_values(by=['lt10_prob'],ascending=True)[:5]

predict.ix[lowerfilter.index,:] = predict.ix[lowerfilter.index,:].applymap(lambda x: x if x >= 8 else 8,)
predict.ix[lowerfilter10.index,:] = predict.ix[lowerfilter10.index,:].applymap(lambda x: x if x >= 10 else 10,)

print(predict.describe())
predict.plot()
plt.show()
predict.to_csv("submission_23_end_filter.csv",header=None,index=None)
# print(predict.ix[up12filter.index,:])
# exit()