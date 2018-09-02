require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'
require 'optim'

paths.dofile('myutils.lua')
paths.dofile("model.lua")

default_config = paths.dofile('getconfig.lua')
utils = paths.dofile('modelUtils.lua')
torch.setdefaulttensortype('torch.FloatTensor') 
torch.setnumthreads(1)
torch.manualSeed(0)
cutorch.manualSeedAll(0)

print("Using activation :")
if default_config.activation == "ReLU" then
  default_config.activation = cudnn.ReLU
elseif default_config.activation == "Sigmoid" then
  default_config.activation = cudnn.Sigmoid
else
  default_config.activation = cudnn.Tanh
end
print(default_config.activation())

print("Using GPU %d" % default_config.gpu)
cutorch.setDevice(default_config.gpu)
default_config.batch = 32
default_config.iSz = 224
default_config.oSz = 56
default_config.gSz = 224

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

print("Running...")
DataLoader = dofile("Neomask/DataLoaderNew.lua")
valLoader = DataLoader(default_config,'val')
data = run_dataset(valLoader,20)

paths.dofile("para_naive.lua")
local m = torch.load("exps/ParallelNet/NaiveSGD6/model.t7")
resnet = torch.load("pretrained/resnet-50.t7")
crit = nn.CrossEntropyCriterion():cuda()
--[[
default_config.model = resnet:clone()
MNet = nn.ParallelNet_Mod(default_config)
default_config.model = None
MNet:make_nets()
MNet:DataParallel(2)
]]
optimState = {}
optimState.learningRate = 0.01
params, grads = resnet:getParameters()
function feval() return crit.output, grads end
resnet:training()
im = data[1].inputs
lbl = data[1].labels

a = cutorch.createCudaHostTensor()
l = a:resize(im:size()):copy(im)
resnet:apply(function(m) m.gradInput=nil end)
nn.ParallelNet.shareGradInput(resnet,"torch.CudaTensor")
resnet = utils.makeDataParallelTable(resnet, 2)
resnet = resnet:cuda()
for i=1,5 do
    print(i)
	res = resnet:forward(a,3)
	loss = crit:forward(res,lbl:cuda())
	gradin = crit:backward(res,lbl:cuda())
	fake_g = resnet:backward(a,gradin,3)
	optim.sgd(feval, params, optimState)
	print(loss)
end
