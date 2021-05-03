import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.dirname(__file__))

def main():
    '''Visualizes distribution of clauses over clusters'''
    sizes = pd.read_csv(dirname + '/output/clusters/finaltopic_sizes.csv', usecols=['Topic', 'Count'], dtype=int)
    sizes = sizes.rename(columns={'Topic': 'Clusters', 'Count': 'Size'}, inplace=False)
    #sizes = sizes.drop(0) #to drop outliers
    sizes['Size'] = sizes['Size'].replace(['old value'], 'new value')
    sns.set_theme()
    f = sns.barplot(x = sizes['Clusters'], y = sizes['Size'], color="#328da8", order=sizes.sort_values('Size',ascending = False).Clusters) #data=sizes, palette=["#32a8a8", "#3273a8"])
    plt.title("Distribution of cluster sizes", size=12)
    f.set(xticklabels=[])
    plt.show()

if __name__ == '__main__':
    main()