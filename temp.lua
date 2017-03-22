require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'

paths.dofile('myutils.lua')
paths.dofile("model.lua")
default_config = paths.dofile('getconfig.lua')
local utils = paths.dofile('modelUtils.lua')

-- set GPU
print("Using GPU %d" % default_config.gpu)
cutorch.setDevice(default_config.gpu)

print("Loading Imagenet List...")
f = io.open("Neomask/synset_words.txt")
syn_list = {}
for i=1,1000 do
	c = f:read()
	syn_list[i] = c:split(",")[1]
end

DataLoader = dofile("Neomask/DataLoaderNew.lua")
trainLoader, valLoader = DataLoader.create(default_config)

print("Running...")
function run_dataset(DL,s)
	local cnt = 0
	local data = {}
	for n, sample in DL:run() do
		if sample ~= nil then
			cnt = cnt + 1
			data[cnt] = sample
		end

		if cnt > s then
			break
		end
	end
	return data
end

data = run_dataset(trainLoader,20)
im = data[1].inputs
masks = data[1].labels
cutorch.setDevice(default_config.gpu)
ins = im:cuda():clone()


--[[
im = torch.CudaTensor()
im:resize(data[1].inputs:size()):copy(data[1].inputs)
print(im:size())

-- forward
pred = resnet:forward(im)
conf,ind = torch.max(pred, 2) -- max decomposition
resnet:zeroGradParameters()

-- build mode matrix
ind = ind:int()
dectar = torch.zeros(im:size(1), 1000)
for i=1,im:size(1) do dectar[i][ind[i][1] ] = 1 end

-- take gradient and relevance
resnet:zeroGradParameters()
gd = resnet:backward(im,dectar:cuda()):clone()
rd = torch.cmul(gd,im)
resnet:zeroGradParameters()

gf = resnet:backward(im,pred):clone()
rf = torch.cmul(gf,im)
prefix = '../res/'

thr = 0.7
for i=1,5 do
	local title = syn_list[ind[i][1] ]
	save_comp(im[i],gd[i],(prefix.."gd"..i..title)..".png",thr,true)
	save_comp(im[i],gf[i],(prefix.."gf"..i..title)..'.png',thr,true)
	save_comp(im[i],rd[i],prefix..i.."rd.png",thr,true)
	save_comp(im[i],rf[i],prefix..i.."rf.png",thr,true)
	print("Distance of 1-hot and full : %.2f" % torch.dist(rd[i],rf[i]))
end
]]--

--[[
require 'tds'
coco = require 'coco'
cfile = coco.CocoApi("/home/atlantix/COCO/COCO/data/annotations/instances_val2014.json")

function eval(item)
	local imgId = item['image_id']
	local seg = item['segmentation']['counts']
	local h = item['segmentation']['size'][1]
	local w = item['segmentation']['size'][2]
	
	local annId = cfile:getAnnIds(imgId)
	return cfile:loadAnns(annId)
end
]]