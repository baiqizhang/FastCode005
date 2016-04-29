require 'paths';
require 'nn';
require 'image';

N = 0
names = {}
for line in io.lines() do
    N=N+1
    names[N] = line
end

--data = torch.Tensor(torch.LongStorage({N,3,32,32}))
--label = torch.Tensor(torch.LongStorage({N}))

--print(testset)

export = torch.load('oxbuild.dat','ascii')

net = export.net
mean = export.mean
stdv = export.stdv

-- Evaluate mode
net:evaluate()

classes = {'all_souls', 'ashmolean', 'christ_church', 'hertford',
           'radcliffe_camera', 'distractor'}

for i=1,N do
    f = names[i]
    img = image.load('/home/hadoop/'..f,3,'byte')
    img = image.scale(img,'^32')
    img = image.crop(img,'c',32,32)
    data = img:double()
    class = f:sub(1,2)
    if (class == 'al') then
        label = 1
    end
    if (class == 'as') then
        label = 2
    end
    if (class == 'ch') then
        label = 3
    end
    if (class == 'he') then
        label = 4
    end
    if (class == 'ra') then
        label = 5
    end
    prediction = net:forward(data):exp()
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    print(indices[1]..'\t'..confidences[1]..' '..names[i])
    i = i+1

end

-- print(#data)
-- print(#label)
-- -- print(label)

--testset = {}
--testset.data = data
--testset.label = label
           
--testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
--for i=1,3 do -- over each image channel
--    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
--    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
--end

--net:remove(#net.modules)
--net:remove(#net.modules)


