import matplotlib.pyplot as plt
import pandas


def plot_accuracy(df_list) :
	# title font
	linestyle_list = ['-', ':', '-.', '--']
	marker_list = ['.', 'o', 'v', 's']
	color_list=['r', 'b', 'g', 'y']
	training_label_list=['Training Acc ','Training Acc Early Stopping', 'Training Loss LeakyReLU', 'Training Loss PReLU']
	validation_label_list=['Validation Acc ', 'Validation Acc Early Stopping','Validation Loss LeakyReLU','Validation Loss PReLU']
	training_loss_label_list=['Training Loss ', 'Training Loss Early Stopping','Training Loss LeakyReLU', 'Training Loss PReLU']
	validation_loss_label_list=['Validation Loss ', 'Validation Loss Early Stopping','Validation Loss LeakyReLU','Validation Loss PReLU']
	title_font = {'family' : 'Times New Roman', 'weight' : 'bold', 'size' : 18, }
	# lable_font
	lable_font = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 12, }
	# epoch = list(range(1, len(list(df_list[0]['training_accuracy'])) + 1, 5))
	for i in range(len(df_list)) :
		df = df_list[i]
		
		acc = list(df['training_accuracy'])[0::5]
		val_acc = list(df['validation_accuracy'])[0::5]
		epoch = list(range(1, len(list(df['training_accuracy'])) + 1, 5))
		
		print(len(epoch), len(acc))
		plt.plot(epoch, acc, color=color_list[i], label=training_label_list[i], linestyle=linestyle_list[0],
		         linewidth=1, markersize=1)
		plt.plot(epoch, val_acc, color=color_list[i], label=validation_label_list[i], linestyle=linestyle_list[1],
		         linewidth=1, markersize=1)
	plt.title('Training and validation accuracy', fontdict=title_font)
	plt.legend(prop=lable_font)
	plt.xlabel("Epoch", fontdict=lable_font)
	plt.ylabel("Accuracy", fontdict=lable_font)
	plt.figure()
	
	for i in range(len(df_list)) :
		df = df_list[i]
		loss = list(df['training_loss'])[0::5]
		val_loss = list(df['validation_loss'])[0::5]
		epoch = list(range(1, len(list(df['training_accuracy'])) + 1, 5))
		plt.plot(epoch, loss, color=color_list[i], label=training_loss_label_list[i], linestyle=linestyle_list[0],
		         linewidth=1, markersize=1)
		plt.plot(epoch, val_loss, color=color_list[i], label=validation_loss_label_list[i], linestyle=linestyle_list[1],
		         linewidth=1, markersize=1)
	plt.title('Training and validation loss', fontdict=title_font)
	plt.legend(prop=lable_font)
	plt.xlabel("Epoch", fontdict=lable_font)
	plt.ylabel("Loss", fontdict=lable_font)
	plt.show()


acc_df=pandas.read_excel("training_record.xlsx")
acc_df0 = pandas.read_excel("training_record_0.xlsx")
acc_df1 = pandas.read_excel("training_record_1.xlsx")
acc_df2 = pandas.read_excel("training_record_2.xlsx")

df_list = []
df_list.append(acc_df0)
df_list.append(acc_df1)
df_list.append(acc_df2)
# df_list.append(acc_df3)
plot_accuracy(df_list)
