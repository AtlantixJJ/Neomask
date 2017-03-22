require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'io'
require 'cutorch'
require 'image'

paths.dofile('myutils.lua')

local default_config = paths.dofile('getconfig.lua')

-- fixing arguments
print("Selecting layer to be 6")
default_config.layer = 6
print("Classification Mode...")
default_config.classify = true

local select_model =  "blgx_fbc_naive.lua"
paths.dofile(select_model)

local utils = paths.dofile('modelUtils.lua')

-- set GPU
default_config.gpu2 = default_config.gpu
default_config.gpu1 = default_config.gpu
print("Using GPU %d" % default_config.gpu)
cutorch.setDevice(default_config.gpu)

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
  denet = nn.DecompNet(default_config,default_config.layer)
  default_config.model = None
end
-- Loading Trainer
paths.dofile('TrainerDBNet.lua')
-- paths.dofile("TrainerSharpMask.lua")
cutorch.setDevice(default_config.gpu2)
local criterion = nn.SoftMarginCriterion():cuda()
local trainer = Trainer(denet, criterion, default_config)

print("Loading Data...")
local DL = paths.dofile('DataLoaderNew.lua')
local TrainDL, ValDL = DL.create(default_config)

epoch=1
print('| start training')
for i = 1, 100 do
  -- if i%2 == 1 then trainer:test(epoch,ValDL) end
  trainer:train(epoch,TrainDL)

  epoch = epoch + 1
end
