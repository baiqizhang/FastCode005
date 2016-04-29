require 'image'
require 'paths'
N = 0
for f in paths.files('oxbuild/training/') do
    if f:sub(1,1)~='.' then
        N=N+1
    end
end

data = torch.Tensor(torch.LongStorage({N,3,32,32}))
label = torch.Tensor(torch.LongStorage({N}))

i = 1
for name in paths.files('oxbuild/training/') do
    if name:sub(1,1)~='.' then
        img = image.load('oxbuild/training/'..name,3,'byte')
        img = image.scale(img,'^32')
        img = image.crop(img,'c',32,32)
        data[i] = img
        label[i] = tonumber(string.split(name,'|')[2]:sub(1,1))
        i = i+1
    end
end


print(#data)
print(#label)
-- print(label)

trainset = {}
trainset.data = data
trainset.label = label
print(trainset)


N = 0
for f in paths.files('oxbuild/classification/') do
    if f:sub(1,1)~='.' then
        N=N+1
    end
end

data = torch.Tensor(torch.LongStorage({N,3,32,32}))
label = torch.Tensor(torch.LongStorage({N}))

i = 1
for f in paths.files('oxbuild/classification/') do
    if f:sub(1,1)~='.' then
        img = image.load('oxbuild/classification/'..f,3,'byte')
        img = image.scale(img,'^32')
        img = image.crop(img,'c',32,32)
        data[i] = img
        class = f:sub(1,2)
        if (class == 'al') then
            label[i] = 1
        end
        if (class == 'as') then
            label[i] = 2
        end
        if (class == 'ch') then
            label[i] = 3
        end
        if (class == 'he') then
            label[i] = 4
        end
        if (class == 'ra') then
            label[i] = 5
        end
        i = i+1
    end
end


-- print(#data)
-- print(#label)
-- -- print(label)

testset = {}
testset.data = data
testset.label = label
print(testset)



classes = {'all_souls', 'ashmolean', 'christ_church', 'hertford',
           'radcliffe_camera', 'distractor'}
print(trainset)
print(#trainset.data)
print(classes[trainset.label[20]])

print(classes[testset.label[40]])

-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

mean = {0,0,0} -- store the mean, to normalize the test set in the future
stdv = {1,1,1} -- store the standard-deviation for the future
-- mean = {}
-- stdv = {}
-- for i=1,3 do -- over each image channel
--     mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
--     print('Channel ' .. i .. ', Mean: ' .. mean[i])
--     trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
--     stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
--     print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
--     trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
-- end
--
require 'nn';
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity 
net:add(nn.Linear(84, 6))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems
-- net = net:cuda() --use cuda

criterion = nn.ClassNLLCriterion() --Loss function
-- criterion = criterion:cuda() --use cuda

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 15 -- just do 5 epochs of training.
trainer:train(trainset)



testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
i = 93
print(classes[testset.label[i]])
predicted = net:forward(testset.data[i])

predicted:exp()
for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

correct = 0
for i=1,100 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print(correct, correct .. ' % ')

class_performance = {0, 0, 0, 0, 0, 0}
for i=1,100 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end
for i=1,#classes do
    print(classes[i], class_performance[i])
end


export = {}
export.net = net
export.mean = mean
export.stdv = stdv
torch.save('oxbuild.dat', export,'ascii')
