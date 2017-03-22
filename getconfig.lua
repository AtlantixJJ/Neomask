local cmd = torch.CmdLine()
cmd:text()
cmd:text('train DeepMask or SharpMask')
cmd:text()
cmd:text('Options:')
cmd:option("-name","Net")
cmd:option('-rundir', 'exps/', 'experiments directory')
cmd:option('-datadir', 'COCO/data/', 'data directory') -- changes
cmd:option('-seed', 1, 'manually set RNG seed')
cmd:option('-gpu', 1, 'gpu device')
cmd:option('-gpu1', 1, 'multi-gpu device')
cmd:option('-gpu2', 2, 'multi-gpu device')
cmd:option('-optim', "adam", 'adam,adadelta,sgd...')
cmd:option('-nthreads', 2, 'number of threads for DataSampler')
cmd:option('-reload','','reload a network from given directory')
cmd:text()
cmd:text('Training Options:')
cmd:option('-batch', 32, 'training batch size')
cmd:option('-layer',1,'from layer')
cmd:option('-classify',false,"if to pre-train classification")
cmd:option("-modeldef","Neomask/blgx_bnaive.lua","LUA model definition file")
cmd:option('-lr', 0, 'learning rate (0 uses default lr schedule)')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-wd', 5e-4, 'weight decay')
cmd:option('-maxload', 4000, 'max number of training batches per epoch')
cmd:option('-testmaxload', 500, 'max number of testing batches')
cmd:option('-maxepoch', 100, 'max number of training epochs')
cmd:option('-iSz', 224, 'input size') -- 192+32=224
cmd:option('-oSz', 56, 'output size')
cmd:option('-gSz', 112, 'ground truth size')
cmd:option('-shift', 16, 'shift jitter allowed')
cmd:option('-scale', .25, 'scale jitter allowed')
cmd:option('-hfreq', 0.5, 'mask/score head sampling frequency')
cmd:option('-scratch', false, 'train DeepMask with randomly initialize weights')
cmd:text()
cmd:text('SharpMask Options:')
cmd:option('-dm', '', 'path to trained deepmask (if dm, then train SharpMask)')
cmd:option('-km', 32, 'km')
cmd:option('-ks', 32, 'ks')

local config = cmd:parse(arg)

-- sharpmask config
config.hfreq = 0 -- train only mask head
config.gSz = config.iSz -- in sharpmask, ground-truth has same dim as input

return config
