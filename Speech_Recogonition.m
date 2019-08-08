%% File Location
dataDir ='C:\Users\Abhinav Raj\AppData\Local\Temp\an4\wav\flacData'
ads=audioDatastore(dataDir,'IncludeSubfolders',true,'FileExtensions','.flac','LabelSource','foldernames')
%% Splitting the data into train & test
[trainDatastore,testDatastore]=splitEachLabel(ads,0.80)
%% counting each label
trainDatasotrecount=countEachLabel(trainDatastore)
testDatastorecount=countEachLabel(testDatastore)
%% Computing MFCC and Pitch
lenDataTrain=length(trainDatastore.Files)
features=cell(lenDataTrain,1)
for i=1:lenDataTrain
    [dataTrain,infoTrain]=read(trainDatastore)
    features{i}=HelperComputePitchAndMFCC(dataTrain,infoTrain)
end
features=vertcat(features{:})
features=rmmissing(features)
head(features)
%% Normalizing the data
featureVectors=features{:,2:15};
m=mean(featureVectors)
s=std(featureVectors)
features{:,2:15}= (featureVectors-m)./s
head(features)
%% Training a classifier
inputTable=features;
predictorNames=features.Properties.VariableNames
predictors=inputTable(:,predictorNames(2:15))
response=inputTable.Label
trainedClassifier=fitcknn(predictors,response,'Distance','euclidean','NumNeighbors',5,'DistanceWeight','squaredinverse','Standardize',false,'ClassNames',unique(response))
%% 5 fold stratified Cross-Validation
k=5
group=(response)
c=cvpartition(group,'KFold',k)
partitionedModel=crossval(trainedClassifier,'CVPartition',c)
%% Compute Validation accuracy
ValidationAccuracy=1-kfoldLoss(partitionedModel,'LossFun','ClassifError')
fprintf('\n Validation Accuracy = %.2f%%\n',ValidationAccuracy*100)
%% Confussion Matrix
ValidationPredictions=kfoldPredict(partitionedModel) 
figure;
cm=confusionchart(features.Label,ValidationPredictions,'title','Validation Accuracy')
cm.ColumnSummary= 'column-normalized'
cm.RowSummary= 'row-normalized'
%% Testing The Classifier
lenDataTest=length(testDatastore.Files)
featuresTest=cell(lenDataTest,1)
for i=1:lenDataTest
    [dataTest,infoTest]=read(testDatastore)
    featuresTest{i}=HelperComputePitchAndMFCC(dataTest,infoTest)
end
featuresTest=vertcat(featuresTest{:})
featuresTest=rmmissing(featuresTest)
featuresTest{:,2:15}=(featuresTest{:,2:15}-m)./s
head(featuresTest)
%% Prediction
result= HelperTestKNNClassifier(trainedClassifier,featuresTest)
figure;
confusionchart(result.ActualSpeaker,result.PredictedSpeaker,'title','Confusion Matrix for Test Data')