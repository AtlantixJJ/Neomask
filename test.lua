require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'
require 'gnuplot'

paths.dofile('myutils.lua')

local select_model = "blg6naive.lua" -- "model.lua"
paths.dofile(select_model)

local default_config = paths.dofile('getconfig.lua')
default_config.reload = "exps/BL_G6_Naive"
local utils = paths.dofile('modelUtils.lua')

-- set GPU
default_config.gpu = 4
print("Using GPU %d" % default_config.gpu)
cutorch.setDevice(default_config.gpu)

print("Loading Imagenet List...")
f = io.open("Neomask/synset_words.txt")
syn_list = {}
for i=1,1000 do
	c = f:read()
	syn_list[i] = c:split(",")[1]
end

local epoch = 1, denet
if #default_config.reload > 0 then
  print("Building from DeNet pretrained model...")
  epoch = 1
  if paths.filep(default_config.reload..'/log') then
    for line in io.lines(default_config.reload..'/log') do
      if string.find(line,'train') then epoch = epoch + 1 end
    end
  end
  print(string.format('| reloading experiment %s', default_config.reload))
  local m = torch.load(string.format('%s/model.t7', default_config.reload))
  denet, config = m.model, m.config
else
  print("Building from ResNet...")
  local resnet = torch.load("pretrained/resnet-50.t7")
  utils.BNtoFixed(resnet, true)
  resnet:add(nn.SoftMax())
  resnet = resnet:cuda()

  local mconfig = {}
  mconfig.batch = default_config.batch
  mconfig.gSz = default_config.gSz
  mconfig.model = resnet
  mconfig.name = default_config.name

  -- Building DeNet
  denet = nn.DecompNet(mconfig)
end

local DataLoader = paths.dofile("DataLoaderNew.lua")
local trainLoader, valLoader = DataLoader.create(config)

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
-- denet.tail:get(2).bias[1] = -2
-- cutorch.synchronize()
c = denet.gradFit.modules[2]
c.bias[1] = 0
res = denet:forward(im:cuda())
res = res:reshape(res:size(1),224,224)


for i=1,16 do
  save_pred(im[i],{res[i]},"res/"..i..".png",3)
end

--[[
for i=4,7 do
  print(i)
  csres = denet:forward(im:cuda(),i)
  for k=1,32 do
    res[1] = csres[1][k]
    res[2] = csres[2][k]
    res[3] = csres[3][k]
    save_pred(im[k],res,"res/"..k.."_"..i..".png")
  end
end

print("Saving...")
for i=1,32 do
  save_comp(im[i], res[i], i..".png", 0.1, true)
end
]]