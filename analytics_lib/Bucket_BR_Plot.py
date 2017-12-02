#plot the number of booked and unbooked conversations with booking rate overlay
#requires the input data has a "booked/has_stay" column
#plot_df: dataframe containing feature set and if/not booked, need to be 1-1
#fea_col: column name of the feature
#booked_col: column name of if/not booked
#bin: number of bins in the histogram
#fig_name: name of the figure that need to be saved, will also be the title of the figure
import numpy
from matplotlib import pyplot
import os
import numpy
from matplotlib import pyplot
import os

def Bucket_BR_Plot(plot_df,fea_col,booked_col,no_bin=None,fig_name=None,ylim=None):
    temp = plot_df[[fea_col,booked_col]]
    temp.columns = ['fea', 'booked']

    
    ax1 = pyplot.subplot() 
    if no_bin is None:
        no_bin = 10
    
    
    h, bins,patches = ax1.hist((temp.iloc[:,0][temp.booked == True],temp.iloc[:,0][temp.booked == False]),no_bin,normed=False)
    
    ax1.legend(['booked','not booked'])

    h_ratio = h[0].astype(float)/(h[1]+h[0]).astype(float)

    h_ratio[numpy.isnan(h_ratio)]=0
    ax1.set_xlim([0, 1])
    if ylim is not None:
        ax1.set_ylim([0, ylim])
    
    ax2 = ax1.twinx()
    
    ax2.scatter((bins[0:-1]+bins[1:])/2.0,h_ratio, c='r')

    if fig_name is None:
        ax1.set_title('testing plot')
    else:
        ax1.set_title(os.path.basename(fig_name).replace('.png',''))
        if '.png' not in fig_name:
            fig_name = fig_name+'.png'


        pyplot.savefig(fig_name)
    pyplot.show()
    pyplot.close()
    return h_ratio

def c_bucketplot(plot_df,fea_col,booked_col,no_bin,fig_name):

    temp = plot_df[[fea_col,booked_col]]
    temp.columns = ['fea', 'booked']


    ax1 = pyplot.subplot()

    h, bins, patches = ax1.hist((temp.iloc[:,0][temp.booked == True],temp.iloc[:,0][temp.booked == False]),no_bin,normed=False)

    ax1.legend(['booked','not booked'])

    h_ratio = h[0].astype(float)/(h[1]+h[0]).astype(float)
    h_ratio[numpy.isnan(h_ratio)]=0

    ax2 = ax1.twinx()


    ax2.scatter((bins[0:-1]+bins[1:])/2.0, h_ratio)
    ax2.plot(numpy.arange(0,1.1,.1),numpy.arange(0,1.1,.1))

    ax2.set_ylim([0,1])
    ax2.set_title(fig_name)

    pyplot.show()
    pyplot.close()
    return h_ratio
