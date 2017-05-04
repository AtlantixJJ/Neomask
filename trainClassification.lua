require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'

paths.dofile('myutils.lua')

local select_model ="class_model.lua" -- "blgx_bnaive.lua" -- "blg6fbc.lua" -- "blg6fbcnaive.lua" -- -- "blg6naive.lua" -- "model.lua"
paths.dofile(select_model)

local default_config = paths.dofile('getconfig.lua')
local utils = paths.dofile('modelUtils.lua')

-- set GPU
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

  default_config.model = resnet
  -- Building DeNet
  denet = nn.ClassNet(default_config,default_config.layer)
  default_config.model = None
end

paths.dofile("TrainerClassification.lua")
cutorch.setDevice(default_config.gpu2)
local trainer = Trainer(denet, default_config)

print("Loading Data...")
local DL = paths.dofile('DataLoaderNew.lua')
local TrainDL,ValDL = DL.create(default_config)



epoch=1
print('| start training')
for i = 1, 100 do
  
  trainer:train(epoch,TrainDL)
  if i%2 == 0 then trainer:test(epoch,ValDL) end
  -- print("Validation:")
  
  epoch = epoch + 1
end
