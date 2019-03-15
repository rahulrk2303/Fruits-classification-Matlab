
alex = alexnet;
layers = alex.Layers

layers(23) = fullyConnectedLayer(6);
layers(25) = classificationLayer

allImages = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainingImages, testImages] = splitEachLabel(allImages, 0.8, 'randomize');

opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 20, 'MiniBatchSize', 64);
myNet = trainNetwork(trainingImages, layers, opts);

predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

FruitsNN = myNet;
save FruitsNN;
