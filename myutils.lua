require 'image'
local coco = require 'coco'
local maskApi = coco.MaskApi

local jet = image.y2jet(torch.linspace(1,1000,1000))

function norm(im)
    for i=1,im:size(1) do
        im[i] = (im[i] - im[i]:min()) / (im[i]:max() - im[i]:min())
        im[i] = im[i] - im[i]:mean()
    end
    return im
end

function get_prediction(batch,net,list,k)
    pred = net:forward(batch)
    conf,ind = torch.max(pred,2)
    res = {}
    for i=1,batch:size(1) do
        local temp = pred[i]
        res[i] = {}
        for j=1,k do
            local c,f = torch.max(temp,1)
            res[i][j] = {f[1],list[f[1]],list[f[1]-1],list[f[1]+1],c[1]}
            temp[f[1]] = 0
        end
    end
    return res
end

function norm_proc(raw,mode)
    local pred = raw:float():clone()
    if pred:size(1) == 1 then
        pred = pred[1]
    end
    if mode == 1 then -- normalize mode
        pred = (pred-pred:min())/(pred:max()-pred:min())
    elseif mode == 2 then -- abs mode
        pred = torch.abs(pred)
        pred = (pred-pred:min())/(pred:max()-pred:min())
    elseif mode == 3 then -- truncate mode
        print(pred:min(),pred:max())
        pred[ torch.le(pred,-1) ] = -1
        pred[ torch.le(-pred,-1)] = 1
        if pred:max() ~= pred:min() then
            pred = (pred-pred:min())/(pred:max()-pred:min())
        end
        print(pred:min(),pred:max())
    elseif mode == 4 then -- to gray
        pred[ torch.le(pred,-1) ] = -1
        pred[ torch.le(-pred,-1)] = 1
        if pred:max() ~= pred:min() then
            pred = (pred-pred:min())/(pred:max()-pred:min())
        end
    end
    return pred
end

function gray2jetrgb(rele)
    -- can be multidimension, but normalization must be done outside

    if rele:nDimension() == 3 then
        -- rgb image => gray first
        nres = image.rgb2y(rele)
    elseif rele:nDimension() > 3 then
        nres = torch.sum(rele,1)
    else 
        nres = rele
    end

    local rgbim = torch.zeros(3,nres:size(1),nres:size(2))

    for i=1,nres:size(1) do
        for j=1,nres:size(2) do
            local ind = torch.floor(nres[i][j]*1000)
            if ind <= 0 then
                ind = 1
            end
            rgbim[{{},i,j}] = jet[{{},1, ind}]
        end
    end
    return rgbim
end

function concate_imgs(imgs)
    local block = torch.zeros(3,imgs:size(3),3):float()
    local res = block:clone():float()
    imgs = imgs:float()
    for i=1,imgs:size(1) do
        -- print(res:size(),imgs[i]:size(),block:size())
        res = torch.cat( torch.cat(res, imgs[i], 3), block ,3)
    end
    return res
end

function save_pred(im,reles,name,mode)
    im = norm_proc(im,1)
    local comb = im:clone():float()
    -- reles is table
    for i=1,#reles do
        rele = gray2jetrgb(norm_proc(reles[i],mode)) -- truncate mode
        rele = rele:float()
    
        comb = torch.cat(comb, rele, 3)
    end
    image.save(name,comb)
end

function show_mask(im,mask,name)
    maskApi.drawMasks(im, mask, 1)
    image.save(name..".png",im)
end

function comp_pred(im,mask,pred,name)
    maskApi.drawMasks(im, mask, 1)
    save_comp(im,pred,name)
end