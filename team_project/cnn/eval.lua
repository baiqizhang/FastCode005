require 'paths';
require 'nn';
export = torch.load('cifarnet.dat','ascii')

net = export.net
mean = export.mean
stdv = export.stdv

classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

testset = torch.load('cifar10-test.t7') 
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

net:remove(#net.modules)
net:remove(#net.modules)

-- Evaluate mode
net:evaluate()

i = 93
output = net:forward(testset.data[i])
print(output)
