--[[
    Forward and backward combined.
]]--

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'cutorch'
local utils = paths.dofile('modelUtils.lua')
if nn.SpatialSymmetricPadding == nil then
    paths.dofile('SpatialSymmetricPadding.lua')
end
local ParallelNet,_ = torch.class("nn.ParallelNet",'nn.Container')

function ParallelNet:set_config(config)
    self.gpu1 = config.gpu1
    self.gpu2 = config.gpu2
    assert(self.gpu1==self.gpu2)
    self.traceback_depth = config.layer - 1

    self.config = config

    self.name = config.name
    self.softmax = nn.SoftMax():cuda()
end

function ParallelNet:__init(config)
    print("Parallel Model on traceback depth %d ." % config.layer)
    self:set_config(config)
    cutorch.setDevice(config.gpu1)
    

    -- the preset model cannot be cuda model
    self.M = config.model
    self.M:remove();self.M:remove();self.M:remove();self.M:remove();
    
    --local temp = self.M:forward(torch.rand(1, 3, self.config.iSz, self.config.iSz):cuda())
    self.fSz = 14
    self.nfeat = 1024
    print("Ground Truth Size : ", self.config.gSz)
    ---- Determine trunk-out shape (rele-origin)

 
    -- 4*remove : start size is (1024,14,14)
    -- 3*remove : start size is (2048, 7, 7)

    --- Build Trunk Head
    --- Trunk head is the common layer of classification and score
    self:build_trunk()
    self:build_classBranch()
    self:build_scoreBranch()

    -- pre-running to determine shapes
    self:precheck_class()
    self:build_maskBranch()
    self:precheck_mask()

    self.classNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.classBranch)
    self.scoreNet = nn.Sequential():add(self.trunk):add(self.trunk_head):add(self.scoreBranch)
    self.maskNet  = nn.Sequential():add(self.trunk):add(self.maskBranch)
    collectgarbage()

    self:make_NewActivation(self.config.activation) 
    self:check()
    -- trunk -> trunk_head -> class/score
    -- trunk -> (new_act) -> maskBranch
end

function ParallelNet:check()
    local data = torch.rand(1, 3, self.config.iSz, self.config.iSz):cuda()
    local output, grad
    print("Checking Mask Branch")
    print("UseBranch : true, UseAct false")
    self:training()
    output = self:forward(data, 3, true, true, false)
    grad = self:backward(data, output, 3, true, false)
    output = self:forward(data, 1, true, true, false)
    grad = self:backward(data, output, 1, true, false)
    print("UseBranch : false, UseAct false")
    output = self:forward(data, 1, true, false, false)
    grad = self:backward(data, output, 1, false, false)   
    print("UseBranch : true, UseAct true")
    output = self:forward(data, 1, true, true, true)
    grad = self:backward(data, output, 1, true, true)       
    print("UseBranch : false, UseAct false")
    output = self:forward(data, 1, true, false, false)
    grad = self:backward(data, output, 1, false, false)   
    self:zeroGradParameters()
    print("Passed")
end

function ParallelNet:print_model()
    print("TRUNK HEAD:")
    print(self.trunk_head)
    print("CLASS")
    print(self.classBranch)
    print("SCORE")
    print(self.scoreBranch)
    print("MASK")
    print(self.maskBranch)
end

function ParallelNet:build_scoreBranch()
    self.scoreBranch = nn.Sequential():add(nn.Linear(1024,1)):cuda()
end

function ParallelNet:build_classBranch()
    self.classBranch = nn.Sequential():add(nn.Linear(1024,90)):cuda()
    --- For convenience of backward propagation
    self.trunk_head_class = nn.Sequential():add(self.trunk_head):add(self.classBranch)
end

function ParallelNet:build_trunk()
    self.trunk_head = nn.Sequential()
    for i=1,self.traceback_depth do
        self.trunk_head:add(self.M.modules[#self.M])
        self.M:remove()
    end
    self.trunk_head:add(nn.SpatialSymmetricPadding(1,1,1,1))
    self.trunk_head:add(cudnn.SpatialMaxPooling(3,3,2,2))
    self.trunk_head:add(nn.View(-1,self.nfeat * self.fSz * self.fSz / 4))
    self.trunk_head:add(nn.Linear(self.nfeat * self.fSz * self.fSz / 4, 1024))
    self.trunk_head:add(cudnn.ReLU(true))
    self.trunk_head:add(nn.Dropout(0.75))
    self.trunk_head:add(nn.Linear(1024, 1024))
    self.trunk_head:add(cudnn.ReLU(true))
    self.trunk_head:add(nn.Dropout(0.75))
    self.trunk_head = self.trunk_head:cuda()
    
    --- Trunk is all the common layers
    self.M = self.M:cuda()
    self.trunk = nn.Sequential():add(self.M):cuda()
    self.trunkNet = nn.Sequential():add(self.trunk):add(self.trunk_head)
    collectgarbage()
end

function ParallelNet:build_maskBranch()
    print("FCN MaskBranch")
    self.maskBranch = nn.Sequential()

    --maskBranch:add(nn.Dropout(.5))
    --maskBranch:add(nn.Linear(2048,512))
    --maskBranch:add(nn.Dropout(.5))
    --maskBranch:add(nn.Linear(512,56*56))
    --maskBranch:add(nn.View(self.config.batch, 56, 56))
    local nfeat = self.nfeat
    local dt = 0 -- 2
    -- (1024,14,14)
    self.maskBranch:add(cudnn.SpatialConvolution(nfeat, nfeat/4, 3, 3, 1, 1))
    self.maskBranch:add(cudnn.SpatialBatchNormalization(nfeat/4))
    self.maskBranch:add(cudnn.ReLU(true)) -- (256,12,12)
    self.maskBranch:add(cudnn.SpatialConvolution(nfeat/4, nfeat/16, 3, 3, 1, 1))
    self.maskBranch:add(cudnn.SpatialBatchNormalization(nfeat/16))
    self.maskBranch:add(cudnn.ReLU(true)) -- (64,10,10)    
    self.maskBranch:add(nn.View(-1,nfeat/16*10*10))
    self.maskBranch:add(nn.Linear(nfeat/16*10*10, 512))
    self.maskBranch:add(nn.Linear(512,56*56))
    self.maskBranch:add(nn.View(-1,56,56))
    -- the mask prediction use marginal loss, there should not be activation

    self.maskBranch:add(nn.SpatialUpSamplingBilinear( math.floor(self.config.gSz / 56) ))
    self.maskBranch:add(nn.View(-1,self.config.gSz*self.config.gSz))

    self.maskBranch = self.maskBranch:cuda()
end

-- take relevance and original input as input, output the activation result
-- simulate LSTM forget gate
function ParallelNet:make_NewActivation(activation)
    local input_layer = nn.ParallelTable()
    activation = activation or cudnn.Tanh
    print("Using activation : ")
    print(activation())
    input_layer:add(nn.Identity()) -- nn.CMul(self.from_size[1], self.from_size, self.from_size) -- use multiple
    input_layer:add(nn.Sequential():add(nn.Mul()):add(nn.Add(1,true)):add(activation())) -- use single scale factor
    --input_layer:add(nn.Sequential():add(nn.CMul(self.from_size[1], self.from_size, self.from_size)):add(nn.CAdd(self.from_size[1], self.from_size, self.from_size)):add(activation))
    self.new_act = nn.Sequential():add(input_layer):add(nn.CMulTable()):cuda()
    print(self.new_act)
end

function ParallelNet:change_Activation(new_activation)
    local subnet = self.new_act.modules[1][2]
    if subnet == nil then
        self:make_NewActivation(new_activation)
        print(self.new_act)
    else
        print(subnet)
        subnet.modules[#subnet] = new_activation():cuda()
        print(subnet)
    end
end

function ParallelNet:precheck_class()
    print("Pre-check :")
    print("Input size : ")
    local tempin = torch.rand(1,3,self.config.iSz,self.config.iSz):cuda()
    print(tempin:size())
    
    local trunk_out = self.trunk:forward( tempin )
    print(trunk_out:size())
    print(self.trunk_head_class)
    local pred = self.softmax:forward(self.trunk_head_class:forward(trunk_out))
    self.trunk_head_class:training()
    local rele_input = self.trunk_head_class:c_backward(trunk_out, pred)

    self.in_size = self.config.iSz
    self.scale = math.floor(self.config.gSz / self.config.oSz)

    print("Class Module output : ")
    print(pred:size())
    print("Relevance Origin :", rele_input:size())
    print("Feature Size : ", self.fSz)
    print("Scale factor  %d " % self.scale)
end

function ParallelNet:precheck_mask()
    local tempin = torch.rand(1,3,self.config.iSz,self.config.iSz):cuda()
    local trunk_out = self.trunk:forward( tempin )
    print(self.maskBranch)
    print(trunk_out:size())
    local maskpred = self.maskBranch:forward(trunk_out)
    local grad = self.maskBranch:backward(trunk_out, maskpred)
    print("Pre-check :")
    print("Mask prediction : ")
    print(maskpred:size())
end

function ParallelNet:forward(input_batch,head,iftrain,ifBranch,useAct)
    -- input_batch should be CudaTensor in gpu1
    cutorch.setDevice(self.gpu1)
    self.inputs = input_batch

    if head == 1 then --mask
        if useAct then
            self.trunk_out = self.trunk:forward( self.inputs )
            ----- pred is probability of classes
            self.pred = self.softmax:forward(self.trunk_head_class:forward(self.trunk_out))
            self.trunk_head_class:training()
            self.rele_input = self.trunk_head_class:c_backward(self.trunk_out, self.pred)
            -- Back to original states
            if iftrain == true then
                self.trunk_head_class:training()
            else self.trunk_head_class:evaluate() end

            self.ss_in = {self.trunk_out, self.rele_input}
            self.branch_input =  self.new_act:forward( self.ss_in )
            self.output = self.maskBranch:forward(self.branch_input)
        else
            if ifBranch then
                --print("MaskBranch")
                self.branch_input = self.trunk:forward(self.inputs)
                self.output = self.maskBranch:forward(self.branch_input)
            else 
                self.output = self.maskNet:forward(self.inputs)
            end
        end
    elseif head == 2 then --score
        if ifBranch then
            self.branch_input = self.trunkNet:forward(self.inputs)
            self.output = self.scoreBranch:forward(self.branch_input)
        else
            self.output = self.scoreNet:forward(self.inputs)
        end
    elseif head == 3 then --class
        if ifBranch then
            self.branch_input = self.trunkNet:forward(self.inputs)
            self.output = self.classBranch:forward(self.branch_input)
        else
            self.output = self.classNet:forward(self.inputs)
        end
    end

    return self.output
end

function ParallelNet:backward(input_batch, gradOutput, head, ifBranch, useAct)
    local gradInput
    cutorch.setDevice(self.gpu1)
    if head == 1 then
        if useAct then 
            gradInput = self.maskBranch:backward(self.branch_input, gradOutput)
            gradInput = self.new_act:backward(self.ss_in, gradInput)
            gradInput = gradInput[1]
            if gradInput == nil then
                print("Backward NIL")
            end
            if ifBranch == false then 
                gradInput = self.trunk:backward(input_batch, gradInput)
            end
        else
            if ifBranch then
                -- print(gradOutput:size())
                -- print(self.branch_input:size())
                gradInput = self.maskBranch:backward(self.branch_input, gradOutput)
            else
                gradInput = self.maskNet:backward(input_batch, gradOutput)
            end
        end
        -- the other forward_prop is ignored
    elseif head == 2 then
        if ifBranch then
            gradInput = self.scoreBranch:backward(self.branch_input, gradOutput)
        else
            gradInput = self.scoreNet:backward(input_batch, gradOutput)
        end
    elseif head == 3 then
        if ifBranch then
            gradInput = self.classBranch:backward(self.branch_input, gradOutput)
        else
            gradInput = self.classNet:backward(input_batch, gradOutput)
        end
    end
    return gradInput
end

function ParallelNet:relevance_forward(input_batch)
    self.inputs = input_batch
    local pre = self.ParallelNet:forward(self.inputs)

    if head == 3 then
        self.output = pre
        return self.output
    end

    local N = #self.ParallelNet.modules
    local D = N - self.layer + 1
    pre = pre * 100

    self.M:zeroGradParameters()
    for i=1,D do
        if N == i then
            pre = self.ParallelNet.modules[N-i+1]:backward(self.inputs, pre)
        else
            pre = self.ParallelNet.modules[N-i+1]:backward(self.ParallelNet.modules[N-i].output, pre)
        end
    end

    wd = pre:clone()
    rele = torch.cmul(wd, self.ParallelNet.modules[self.layer-1].output)

    return wd:sum(2), rele:sum(2)
end

function ParallelNet:relevance_visualize(input)
    self.inputs = input
    local pred = self.ParallelNet:forward(self.inputs):clone()
    local softmax = nn.SoftMax():cuda()
    pred = softmax:forward(pred)
    conf,ind = torch.max(pred)
    fwd = pred:clone():zeros()
    fwd[ind[1]]=1

    self.ParallelNet:zeroGradParameters()
    self.full_decomp = self.ParallelNet:backward(input_batch,pred*100)

    self.rele = torch.cmul(self.full_decomp,self.inputs)
    return self.rele:sum(2), self.full_decomp:sum(2)
end

function ParallelNet:training()
    cutorch.setDevice(self.gpu1)
    self.trunk:training()
    self.trunk_head:training()
    self.new_act:training()
    self.maskBranch:training()
    self.classBranch:training()
    self.scoreBranch:training()
end

function ParallelNet:evaluate()
    cutorch.setDevice(self.gpu1)
    self.trunk:evaluate()
    self.trunk_head:evaluate()
    self.new_act:evaluate()
    self.classBranch:evaluate()
    self.maskBranch:evaluate()
    self.scoreBranch:evaluate()
end

function ParallelNet:zeroGradParameters()
    cutorch.setDevice(self.gpu1)
    self.trunk:zeroGradParameters()
    self.trunk_head:zeroGradParameters()
    self.new_act:zeroGradParameters()
    self.classBranch:zeroGradParameters()
    self.maskBranch:zeroGradParameters()
    self.scoreBranch:zeroGradParameters()
end

function ParallelNet:updateParameters(lr)
    cutorch.setDevice(self.gpu1)
    self.trunk:updateParameters(lr)
    self.trunk_head:updateParameters(lr)
    self.new_act:updateParameters(lr)
    self.classBranch:updateParameters(lr)
    self.maskBranch:updateParameters(lr)
    self.scoreBranch:updateParameters(lr)
end

function ParallelNet:cuda()
    cutorch.setDevice(self.gpu1)
    self.trunk:cuda()
    self.trunk_head:cuda()
    self.new_act:cuda()
    self.classBranch:cuda()
    self.maskBranch:cuda()
    self.scoreBranch:cuda()
end

function ParallelNet:float()
    self.trunk:float()
    self.trunk_head:float()
    self.new_act:float()
    self.classBranch:float()
    self.maskBranch:float()
    self.scoreBranch:float()
end

function ParallelNet:clone(...)
    local f = torch.MemoryFile("rw"):binary()
    f:writeObject(self); f:seek(1)
    local clone = f:readObject(); f:close()

    if select('#',...) > 0 then
        clone.trunk:share(self.trunk,...)
        clone.trunk_head:share(self.trunk_head,...)
        clone.new_act:share(self.new_act,...)
        clone.classBranch:share(self.classBranch,...)
        clone.scoreBranch:share(self.scoreBranch,...)
        clone.maskBranch:share(self.maskBranch,...)
    end

    return clone
end

return nn.ParallelNet