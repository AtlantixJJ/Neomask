require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'

paths.dofile('myutils.lua')
paths.dofile("model.lua")
local default_config = paths.dofile('getconfig.lua')
local utils = paths.dofile('modelUtils.lua')

-- set GPU
default_config.gpu = 2
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
-- Loading Trainer
paths.dofile('TrainerDenet.lua')
local criterion = nn.SoftMarginCriterion():cuda()
local trainer = Trainer(denet, criterion, default_config)

print("Loading Data...")
local DL = paths.dofile('DataLoaderNew.lua')
local TrainDL, ValDL = DL.create(default_config)

print('| start training')
for i = 1, default_config.maxepoch do
  trainer:train(epoch,TrainDL)
  print("Validation:")
  if i%2 == 0 then trainer:test(epoch,ValDL) end
  epoch = epoch + 1
end