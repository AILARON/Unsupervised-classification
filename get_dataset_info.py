


from load_dataset import importKaggleTrain, importKaggleTest, importLuoTraining, importPastoreTraining, importAilaronTraining
import numpy as np

AILARON_CLASSES =  ['copepod','diatom_chain','other','faecal_pellets','bubble','oily_gas','oil',""]


class data_info():

    data = None
    labels = None
    num_classes = 121

    def max_min_image_size(self):
        minWidth = self.data[0].shape[0]
        maxWidth = self.data[0].shape[0]
        minHeight = self.data[0].shape[1]
        maxHeight = self.data[0].shape[1]

        for d in self.data:
            if( d.shape[0] > maxWidth):
                maxWidth = d.shape[0]
            elif(d.shape[0] < minWidth):
                minWidth = d.shape[0]
            if( d.shape[1] > maxHeight):
                maxHeight = d.shape[1]
            elif(d.shape[1] < minHeight):
                minHeight = d.shape[1]
        return [minWidth, maxWidth, minHeight, maxHeight]

    def min_image_size(self):
        minWidth = self.data[0].shape[0]
        height = self.data[0].shape[1]

        width = self.data[0].shape[0]
        minHeight = self.data[0].shape[1]

        for d in self.data:
            if( d.shape[0] < minWidth):
                minWidth = d.shape[0]
                height = d.shape[1]
            if( d.shape[1] < minHeight):
                minHeight = d.shape[1]
                width = d.shape[0]

        return [height, minWidth, minHeight, width]

    def max_image_size(self):
        maxWidth = self.data[0].shape[0]
        height = self.data[0].shape[1]

        width = self.data[0].shape[0]
        maxHeight = self.data[0].shape[1]

        for d in self.data:
            if( d.shape[0] > maxWidth):
                maxWidth = d.shape[0]
                height = d.shape[1]
            if( d.shape[1] > maxHeight):
                maxHeight = d.shape[1]
                width = d.shape[0]

        return [height, maxWidth, maxHeight, width]

    def frequencygraph(self):

        ###INFO INFO INFO ###

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(figsize=(10,10))
        plt.subplots_adjust(bottom = 0.22)

        # An "interface" to matplotlib.axes.Axes.hist() method
        n, bins, patches =  axs.hist(x=self.labels, bins=range(self.num_classes +1), color='#0504aa',alpha=0.7, rwidth=0.55, align='left')

        # Set title, grid and axis labels
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Plankton species')
        plt.ylabel('Number of species')
        plt.title('Samples in Ailaron dataset')

        #Change colors >> high freq labels are red
        maxfreq = n.max()
        print(maxfreq)
        cmap = plt.get_cmap('autumn')
        colors = [cmap(max(0, 250-int(i*500/maxfreq))) for i in n]
        for n, color in zip(patches,colors):
            n.set_facecolor(color)

        # now, define the ticks (i.e. locations where the labels will be plotted)
        xticks = [i for i in range(self.num_classes +1)]

        # also define the labels we'll use (note this MUST have the same size as `xticks`!)
        xtick_labels = [f for f in AILARON_CLASSES]

        # add the ticks and labels to the plot
        plt.xticks(rotation='vertical')
        axs.set_xticks(xticks)
        axs.set_xticklabels(xtick_labels)

        plt.savefig('foo.png')

    def number_of_classes(self):
        self.num_classes = len(np.unique(self.labels))
        return len(np.unique(self.labels))

    def number_in_classes(self):
        from collections import Counter
        number_per_class = Counter(self.labels)
        values = number_per_class.values()
        return [max(values),min(values)]

    def data_statistics(self):
        self.data, self.labels = importAilaronTraining(depth=1)



        import tensorflow as tf
        from skimage.transform import resize
        import cv2
        import numpy as np
        from utils import visualize_input


        train_data = self.data #np.array([cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR ) for img in (self.data)])


        visualize_input(train_data[10:50])

        print("Min width, max width, min height, max height")
        print( self.max_min_image_size())

        print("Min width + height, width + min height")
        print(self.min_image_size())

        print("Max width + height, width + max height")
        print(self.max_image_size())

        print(self.number_of_classes())
        print(self.number_in_classes())
        self.frequencygraph()
