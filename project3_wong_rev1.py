
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import backend
#from keras.regularizers import l2
#from keras.regularizers import l1
from keras import regularizers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

layer1 = 25
layer2 = 25
layer3 = 25
layer4 = 30
layer5 = 15
layer6 = 15
output1 = 1
learnRate = 0.001
layer1L2Reg = 0.000001
layer2L2Reg = 0.000001
layer3L2Reg = 0.000001
layer4L2Reg = 0.000001
layer5L2Reg = 0.0000005
L1Reg = 0.0001
dropOut1 = 0.2
dropOut2 = 0.2
dropOut3 = 0.2
epochNumber = 1000
batchSize = 200

modelName = 'NormD-5a'


trainDataSet = "train.csv"
testDataSet = "test.csv"
featureList = ["Recipe_Time", "Clean_Time", "Param1", "NRun", "NSlot", "NGSlot", "NPM"]

featureCount = 7
outPutList = "Tput"

# define base model
adam = optimizers.adam(lr=learnRate)
RMSProp = optimizers.rmsprop(lr=learnRate)
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(layer1, input_dim=featureCount, kernel_initializer='he_uniform', activation='relu'))
    #                kernel_regularizer=regularizers.l2(layer1L2Reg), name='dense_1'))
    #model.add(Dropout(dropOut1))   
    model.add(Dense(layer2, kernel_initializer='he_uniform', activation='relu'))
    #                kernel_regularizer=regularizers.l2(layer2L2Reg), name='dense_2'))
    #model.add(Dropout(dropOut2))
    model.add(Dense(layer3, kernel_initializer='he_uniform', activation='relu'))
    #                kernel_regularizer=regularizers.l2(layer3L2Reg), name='dense_3'))
    #model.add(Dropout(dropOut))
    #model.add(Dense(layer4, kernel_initializer='he_uniform', activation='relu'))
    #                kernel_regularizer=regularizers.l2(layer4L2Reg),
    #                activity_regularizer=regularizers.l1(L1Reg), name='dense_4'))
    #model.add(Dense(layer5, kernel_initializer='he_uniform', activation='relu'))
    #model.add(Dense(layer6, kernel_initializer='he_uniform', activation='relu'))
    #               kernel_regularizer=regularizers.l2(layer5L2Reg),
    #               activity_regularizer=regularizers.l1(L1Reg), name='dense_5'))
    #model.add(Dropout(dropOut3))
    model.add(Dense(units=output1, name='dense_out'))
    #To load weights from previous results stored in a file "weights.best.hdf5" in working directory
    #model.load_weights("weights.best.hdf5")
    #Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    # compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None,
    #         weighted_metrics=None, target_tensors=None)
    # optimizer='adam': Use Adam optimizer to control the speed of learning
    # loss='mean_squared_error': calculates MSE based on y_true and y_pred
    return model
    
ANN_reg = baseline_model()

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
   
# Create Train dataset
tputTrainData = pd.read_csv(trainDataSet)
tputTrainPredictors = tputTrainData[featureList].copy()
tputTrainTargets = tputTrainData[outPutList].copy()
Y_Train = tputTrainTargets.values

# Create Test dataset
tputTestData = pd.read_csv(testDataSet)
tputTestPredictors = tputTestData[featureList].copy()
tputTestTargets = tputTestData[outPutList].copy()
Y_Test = tputTestTargets.values

# Convert predictors to float datatype
X_Train = tputTrainPredictors[featureList].astype(float).values
X_Test = tputTestPredictors[featureList].astype(float).values


# fix random seed for reproducibility
seed = 7

# Instantiate KerasRegressor
#ANN_reg = KerasRegressor(build_fn=baseline_model, epochs=epochNumber, batch_size=batchSize, verbose = 1 )

#Setup checkpoints to capture the best model parameters into a file for re-use later
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
filepath=modelName + "_Best_weights.h5"
csv_logger = CSVLogger(modelName + '_training.log')
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='max')
early_stopping_monitor = EarlyStopping(monitor='loss', min_delta=0.01, patience=100)
callback_list = [checkpoint, csv_logger, early_stopping_monitor]
#callback_list = [checkpoint, csv_logger]

# Train the model using internal data split
history = ANN_reg.fit(X_Train, Y_Train,
                      epochs=epochNumber, 
                      batch_size=batchSize, 
                      verbose = 1, 
                      callbacks=callback_list, 
                      validation_split=0.2,
#                      validation_data(X_Test, Y_Test),
                      shuffle=True)
#pd.DataFrame(history.history).to_csv(modelName + '_history.csv')
plt.plot(history.history['loss'], 'b-')
plt.plot(history.history['val_loss'], 'r-')
plt.axis([0,epochNumber, 0, 2])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.title('model loss')
plt.show()
plt.close()

#Save model
ANN_reg.model.save(modelName + '.h5')
#Save model architecture only to json
json_model = ANN_reg.model.to_json()
open(modelName + '.json', 'w').write(json_model)
#Save model weights only to h5
ANN_reg.model.save_weights(modelName + '_weights.h5')

final_predict = ANN_reg.predict(X_Train)
final_predict_mse = mean_squared_error(Y_Train, final_predict, multioutput = "raw_values")
final_predict_rmse = np.sqrt(final_predict_mse)
print("Train rmse score: ", final_predict_rmse)


final_predict_tr = pd.DataFrame(final_predict, columns=['predictedTput'])
df = pd.concat([tputTrainData, final_predict_tr], axis=1)
#print(df.head())

df1 = df.query('(Slot > 10) & (Slot < 20)').groupby(['Recipe', 'Clean', 'Run', 'JIT1'])['Tput', 'predictedTput'].median().reset_index()
df1.rename(columns={'Tput':'medianTput', 'predictedTput':'predictedMedianTput'}, inplace=True)
print(df1.head())


fig, ax = plt.subplots()
df1.plot(kind="scatter", x="medianTput", y="predictedMedianTput", alpha=0.4, 
          s=(df1["Clean"]/10)**2, label="Median Tput", figsize=(10,7), 
          c="Recipe",cmap=plt.get_cmap("jet"), colorbar=True, edgecolor='black', ax=ax)
plt.show()
plt.close()

Recipe = 150.0
Clean = 60.0
plotMedianTputList = []
legendList = []
ymin = 8
ymax = 18
xmin = 40
xmax = 110

for y in range(1,7):
    plotMedianTput = df1.query('(Recipe==@Recipe) & (Clean==@Clean) & (Run==@y)')
    plotMedianTputList.append(plotMedianTput)
    legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' Run= ' + str(y))
    
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotMedianTputList[0].JIT1.values, plotMedianTputList[0].medianTput.values, 'o-')
ax1.plot(plotMedianTputList[0].JIT1.values, plotMedianTputList[0].predictedMedianTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotMedianTputList[1].JIT1.values, plotMedianTputList[1].medianTput.values, 'o-')
ax2.plot(plotMedianTputList[1].JIT1.values, plotMedianTputList[1].predictedMedianTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotMedianTputList[2].JIT1.values, plotMedianTputList[2].medianTput.values, 'o-')
ax3.plot(plotMedianTputList[2].JIT1.values, plotMedianTputList[2].predictedMedianTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotMedianTputList[3].JIT1.values, plotMedianTputList[3].medianTput.values, 'o-')
ax4.plot(plotMedianTputList[3].JIT1.values, plotMedianTputList[3].predictedMedianTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotMedianTputList[4].JIT1.values, plotMedianTputList[4].medianTput.values, 'o-')
ax5.plot(plotMedianTputList[4].JIT1.values, plotMedianTputList[4].predictedMedianTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotMedianTputList[5].JIT1.values, plotMedianTputList[5].medianTput.values, 'o-')
ax6.plot(plotMedianTputList[5].JIT1.values, plotMedianTputList[5].predictedMedianTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()

Calculate median Tput from slot10 to slot20 as systemTput for each combination of Recipe / Clean / Run / JIT1

corr_matrix = df1.corr()
print(corr_matrix["predictedMedianTput"].sort_values(ascending=False))
fig, ax=plt.subplots()
df1.plot(kind="scatter", x="medianTput", y="predictedMedianTput", alpha=0.4, 
          s=(df1["Clean"]/10)**2, label="Median Tput", figsize=(10,7), 
          c="Recipe",cmap=plt.get_cmap("jet"), colorbar=True, edgecolor='black', ax=ax)
plt.show()
plt.close()


Recipe = 127.0
Clean = 78.0
del plotMedianTputList[:]
del legendList[:]
ymin = 8
ymax = 18
xmin = 40
xmax = 110

for y in range(1,7):
    plotMedianTput = df1.query('(Recipe==@Recipe) & (Clean==@Clean) & (Run==@y)')
    plotMedianTputList.append(plotMedianTput)
    legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' Run= ' + str(y))
    
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotMedianTputList[0].JIT1.values, plotMedianTputList[0].medianTput.values, 'bo-')
ax1.plot(plotMedianTputList[0].JIT1.values, plotMedianTputList[0].predictedMedianTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotMedianTputList[1].JIT1.values, plotMedianTputList[1].medianTput.values, 'bo-')
ax2.plot(plotMedianTputList[1].JIT1.values, plotMedianTputList[1].predictedMedianTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotMedianTputList[2].JIT1.values, plotMedianTputList[2].medianTput.values, 'bo-')
ax3.plot(plotMedianTputList[2].JIT1.values, plotMedianTputList[2].predictedMedianTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotMedianTputList[3].JIT1.values, plotMedianTputList[3].medianTput.values, 'bo-')
ax4.plot(plotMedianTputList[3].JIT1.values, plotMedianTputList[3].predictedMedianTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotMedianTputList[4].JIT1.values, plotMedianTputList[4].medianTput.values, 'bo-')
ax5.plot(plotMedianTputList[4].JIT1.values, plotMedianTputList[4].predictedMedianTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotMedianTputList[5].JIT1.values, plotMedianTputList[5].medianTput.values, 'bo-')
ax6.plot(plotMedianTputList[5].JIT1.values, plotMedianTputList[5].predictedMedianTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()
del legendList[:]
del plotTputList[:]
ymin = 8
ymax = 18
xmin = 0
xmax = 150

JIT1Index = np.sort(df.JIT1.unique())
for x in np.nditer(JIT1Index):
    plotTput = df.query('(JIT1==@x) & (Recipe==@Recipe) & (Clean==@Clean)')
    if x%10 == 0.0: 
        plotTputList.append(plotTput)
        legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' JIT1= ' + str(x))

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].Tput.values, 'bo-')
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].predictedTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].Tput.values, 'bo-')
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].predictedTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].Tput.values, 'bo-')
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].predictedTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].Tput.values, 'bo-')
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].predictedTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].Tput.values, 'bo-')
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].predictedTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].Tput.values, 'bo-')
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].predictedTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].Tput.values, 'bo-')
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].predictedTput.values, 'ro-')
ax7.set_title(legendList[6])
ax7.grid(True)
ax7.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()

Recipe = 175.0
Clean = 88.0
del plotMedianTputList[:]
del legendList[:]
ymin = 9
ymax = 13
xmin = 40
xmax = 110

for y in range(1,7):
    plotMedianTput = df1.query('(Recipe==@Recipe) & (Clean==@Clean) & (Run==@y)')
    plotMedianTputList.append(plotMedianTput)
    legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' Run= ' + str(y))
    
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotMedianTputList[0].JIT1.values, plotMedianTputList[0].medianTput.values, 'bo-')
ax1.plot(plotMedianTputList[0].JIT1.values, plotMedianTputList[0].predictedMedianTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotMedianTputList[1].JIT1.values, plotMedianTputList[1].medianTput.values, 'bo-')
ax2.plot(plotMedianTputList[1].JIT1.values, plotMedianTputList[1].predictedMedianTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotMedianTputList[2].JIT1.values, plotMedianTputList[2].medianTput.values, 'bo-')
ax3.plot(plotMedianTputList[2].JIT1.values, plotMedianTputList[2].predictedMedianTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotMedianTputList[3].JIT1.values, plotMedianTputList[3].medianTput.values, 'bo-')
ax4.plot(plotMedianTputList[3].JIT1.values, plotMedianTputList[3].predictedMedianTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotMedianTputList[4].JIT1.values, plotMedianTputList[4].medianTput.values, 'bo-')
ax5.plot(plotMedianTputList[4].JIT1.values, plotMedianTputList[4].predictedMedianTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotMedianTputList[5].JIT1.values, plotMedianTputList[5].medianTput.values, 'bo-')
ax6.plot(plotMedianTputList[5].JIT1.values, plotMedianTputList[5].predictedMedianTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()
   

del legendList[:]
del plotTputList[:]
ymin = 9
ymax = 13
xmin = 0
xmax = 150

JIT1Index = np.sort(df.JIT1.unique())
for x in np.nditer(JIT1Index):
    plotTput = df.query('(JIT1==@x) & (Recipe==@Recipe) & (Clean==@Clean)')
    if x%10 == 0.0: 
        plotTputList.append(plotTput)
        legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' JIT1= ' + str(x))

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].Tput.values, 'bo-')
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].predictedTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].Tput.values, 'bo-')
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].predictedTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].Tput.values, 'bo-')
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].predictedTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].Tput.values, 'bo-')
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].predictedTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].Tput.values, 'bo-')
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].predictedTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].Tput.values, 'bo-')
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].predictedTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].Tput.values, 'bo-')
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].predictedTput.values, 'ro-')
ax7.set_title(legendList[6])
ax7.grid(True)
ax7.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()
Recipe = 143.0
Clean = 56.0
del plotMedianTputList[:]
del legendList[:]
ymin = 12
ymax = 16
xmin = 40
xmax = 110

for y in range(1,7):
    plotMedianTput = df1.query('(Recipe==@Recipe) & (Clean==@Clean) & (Run==@y)')
    plotMedianTputList.append(plotMedianTput)
    legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' Run= ' + str(y))
    
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotMedianTputList[0].PARM1.values, plotMedianTputList[0].medianTput.values, 'bo-')
ax1.plot(plotMedianTputList[0].PARM1.values, plotMedianTputList[0].predictedMedianTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotMedianTputList[1].PARM1.values, plotMedianTputList[1].medianTput.values, 'bo-')
ax2.plot(plotMedianTputList[1].PARM1.values, plotMedianTputList[1].predictedMedianTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotMedianTputList[2].PARM1.values, plotMedianTputList[2].medianTput.values, 'bo-')
ax3.plot(plotMedianTputList[2].PARM1.values, plotMedianTputList[2].predictedMedianTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotMedianTputList[3].PARM1.values, plotMedianTputList[3].medianTput.values, 'bo-')
ax4.plot(plotMedianTputList[3].PARM1.values, plotMedianTputList[3].predictedMedianTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotMedianTputList[4].PARM1.values, plotMedianTputList[4].medianTput.values, 'bo-')
ax5.plot(plotMedianTputList[4].PARM1.values, plotMedianTputList[4].predictedMedianTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotMedianTputList[5].PARM1.values, plotMedianTputList[5].medianTput.values, 'bo-')
ax6.plot(plotMedianTputList[5].PARM1.values, plotMedianTputList[5].predictedMedianTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()
   
"""
Plot predictedTput against Tput as a function of Global Slot Number for selected Recipe / Clean from Train set
""" 
del legendList[:]
del plotTputList[:]
ymin = 8
ymax = 16
xmin = 0
xmax = 150

PARM1Index = np.sort(df.PARM1.unique())
for x in np.nditer(PARM1Index):
    plotTput = df.query('(PARM1==@x) & (Recipe==@Recipe) & (Clean==@Clean)')
    if x%10 == 0.0: 
        plotTputList.append(plotTput)
        legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' PARM1= ' + str(x))

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].Tput.values, 'bo-')
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].predictedTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].Tput.values, 'bo-')
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].predictedTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].Tput.values, 'bo-')
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].predictedTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].Tput.values, 'bo-')
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].predictedTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].Tput.values, 'bo-')
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].predictedTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].Tput.values, 'bo-')
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].predictedTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].Tput.values, 'bo-')
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].predictedTput.values, 'ro-')
ax7.set_title(legendList[6])
ax7.grid(True)
ax7.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()

"""
Plot predictedMedianTput against targetTput as a function of PARM1 for a given Recipe / Clean / Run
"""
Recipe = 150.0
Clean = 50.0
del plotMedianTputList[:]
del legendList[:]
ymin = 11
ymax = 16
xmin = 40
xmax = 110

for y in range(1,7):
    plotMedianTput = df1.query('(Recipe==@Recipe) & (Clean==@Clean) & (Run==@y)')
    plotMedianTputList.append(plotMedianTput)
    legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' Run= ' + str(y))
    
f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotMedianTputList[0].PARM1.values, plotMedianTputList[0].medianTput.values, 'bo-')
ax1.plot(plotMedianTputList[0].PARM1.values, plotMedianTputList[0].predictedMedianTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotMedianTputList[1].PARM1.values, plotMedianTputList[1].medianTput.values, 'bo-')
ax2.plot(plotMedianTputList[1].PARM1.values, plotMedianTputList[1].predictedMedianTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotMedianTputList[2].PARM1.values, plotMedianTputList[2].medianTput.values, 'bo-')
ax3.plot(plotMedianTputList[2].PARM1.values, plotMedianTputList[2].predictedMedianTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotMedianTputList[3].PARM1.values, plotMedianTputList[3].medianTput.values, 'bo-')
ax4.plot(plotMedianTputList[3].PARM1.values, plotMedianTputList[3].predictedMedianTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotMedianTputList[4].PARM1.values, plotMedianTputList[4].medianTput.values, 'bo-')
ax5.plot(plotMedianTputList[4].PARM1.values, plotMedianTputList[4].predictedMedianTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotMedianTputList[5].PARM1.values, plotMedianTputList[5].medianTput.values, 'bo-')
ax6.plot(plotMedianTputList[5].PARM1.values, plotMedianTputList[5].predictedMedianTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()
   
"""
Plot predictedTput against Tput as a function of Global Slot Number for selected Recipe / Clean from Train set
""" 
del legendList[:]
del plotTputList[:]
ymin = 9
ymax = 16
xmin = 0
xmax = 150

PARM1Index = np.sort(df.PARM1.unique())
for x in np.nditer(PARM1Index):
    plotTput = df.query('(PARM1==@x) & (Recipe==@Recipe) & (Clean==@Clean)')
    if x%10 == 0.0: 
        plotTputList.append(plotTput)
        legendList.append('Recipe= ' + str(Recipe) + ' Clean= ' + str(Clean) + ' PARM1= ' + str(x))

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, sharex='col', sharey='row', figsize=(15,15))
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].Tput.values, 'bo-')
ax1.plot(plotTputList[0].GSlot.values, plotTputList[0].predictedTput.values, 'ro-')
ax1.set_title(legendList[0])
ax1.grid(True)
ax1.set_ylim(ymin, ymax)
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].Tput.values, 'bo-')
ax2.plot(plotTputList[1].GSlot.values, plotTputList[1].predictedTput.values, 'ro-')
ax2.set_title(legendList[1])
ax2.grid(True)
ax2.set_ylim(ymin, ymax)
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].Tput.values, 'bo-')
ax3.plot(plotTputList[2].GSlot.values, plotTputList[2].predictedTput.values, 'ro-')
ax3.set_title(legendList[2])
ax3.grid(True)
ax3.set_ylim(ymin, ymax)
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].Tput.values, 'bo-')
ax4.plot(plotTputList[3].GSlot.values, plotTputList[3].predictedTput.values, 'ro-')
ax4.set_title(legendList[3])
ax4.grid(True)
ax4.set_ylim(ymin, ymax)
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].Tput.values, 'bo-')
ax5.plot(plotTputList[4].GSlot.values, plotTputList[4].predictedTput.values, 'ro-')
ax5.set_title(legendList[4])
ax5.grid(True)
ax5.set_ylim(ymin, ymax)
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].Tput.values, 'bo-')
ax6.plot(plotTputList[5].GSlot.values, plotTputList[5].predictedTput.values, 'ro-')
ax6.set_title(legendList[5])
ax6.grid(True)
ax6.set_ylim(ymin, ymax)
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].Tput.values, 'bo-')
ax7.plot(plotTputList[6].GSlot.values, plotTputList[6].predictedTput.values, 'ro-')
ax7.set_title(legendList[6])
ax7.grid(True)
ax7.set_ylim(ymin, ymax)
plt.legend(['target', 'prediction'], loc='lower right')
plt.show()
plt.close()

