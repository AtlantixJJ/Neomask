require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'
require 'gnuplot'

paths.dofile('myutils.lua')

local select_model = "DeepMask.lua" --blg6fbc.lua" -- "model.lua"
paths.dofile(select_model)

local default_config = paths.dofile('getconfig.lua')
default_config.reload = "model/DMT/"
local utils = paths.dofile('modelUtils.lua')

-- set GPU
default_config.gpu = 3
default_config.gpu2 = default_config.gpu
default_config.gpu1 = default_config.gpu

print("Using GPU %d" % default_config.gpu)
cutorch.setDevice(default_config.gpu)

print("Loading Imagenet List...")
f = io.open("Neomask/synset_words.txt")
syn_list = {}
for i=1,1000 do
	c = f:read()
	syn_list[i] = c:split(",")[1]
end

local epoch = 1
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

  default_config.model = resnet
  -- Building DeNet
  denet = nn.DecompNet(default_config)
  default_config.model = None
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
cutorch.setDevice(default_config.gpu)
ins = im:cuda():clone()

----- for deepmask

mask,score = denet:predict_m(ins)
mask = mask:reshape(mask:size(1),112,112)

print(mask:size(),score:size())

for i=1,32 do
  save_pred(image.scale(im[i],112),{mask[i]},"res/dm"..i..score[i][1]..".png",1)
end