"""
install : pip install visdom
start : python -m visdom.server
observe : open http://111.207.202.17:8097 in your browser
"""

import torch
import visdom

vis = visdom.Visdom(use_incoming_socket=False)
assert vis.check_connection()


# local
# channelMap2compactMap = function(channelMap, flag, n)
# assert (channelMap:dim() == 4)
# assert (channelMap:size(1) == 1)
# n = n or channelMap:size(2)
# assert (n >= channelMap:size(2))
#
# local
# compactMap = torch.zeros(1, 1, channelMap:size(3), channelMap: size(4))
# for k = 1, channelMap:size(2)
# do
# local
# t = channelMap[{{}, {k}, {}, {}}]
# if flag == 0 then
# compactMap[t:gt(0)] = k % n
# else
# compactMap = torch.cmax(compactMap, t * flag)
# end
# end
# return compactMap
# end

def channelMap2compactMap(channelMap, flag, n=None):
    assert channelMap.dim == 4
    assert channelMap.size(1) == 1
    n = n or channelMap.size(2)
    assert n >= channelMap.size(2)
    compactMap = torch.zeros(1, 1, channelMap.size(3), channelMap.size(4))
    for k in range(0, channelMap.size(2)):
        t = channelMap[:, k, :, :]
        # *******
        if flag == 0:
            compactMap[t.gt(0)] = k % n
        else:
            compactMap = torch.cmax(compactMap, t * flag)
    return compactMap


# local
# batch_channelMap2compactMap = function(channelMap, flag)
# assert (channelMap:dim() == 4)
# local
# compactMaps = torch.zeros(channelMap:size(1), 1, channelMap: size(3), channelMap: size(4))
# for i = 1, channelMap:size(1)
# do
# compactMaps[{{i}, {}, {}, {}}] = channelMap2compactMap(channelMap[{{i}, {}, {}, {}}], flag)
# end
# return compactMaps
# end

def batch_channelMap2compactMap(channelMap, flag):
    assert channelMap.dim == 4
    compactMaps = torch.zeros(channelMap.size(1), 1, channelMap.size(3), channelMap.size(4))
    for i in range(0, channelMap.size(1)):
        compactMaps[i, :, :, :] = channelMap2compactMap(channelMap[i, :, :, :], flag)
    return compactMaps


# local
# maxingChannelMap2compactMap = function(channelMap, n) - - assuems that flag is always true
# assert (channelMap:dim() == 4)
# assert (channelMap:size(1) == 1)
# n = n or channelMap:size(2)
# assert (n >= channelMap:size(2))
# local
# compactMap = torch.zeros(1, 1, channelMap:size(3), channelMap: size(4))
# local
# max_indices
# _, max_indices = torch.max(channelMap, 2)
# max_indices = max_indices % n
# return max_indices
# end

def maxingChannelMap2compactMap(channelMap, n=None):  # assuems that flag is always true
    assert channelMap.dim == 4
    assert channelMap.size(1) == 1
    n = n or channelMap.size(2)
    assert n >= channelMap.size(2)
    compactMap = torch.zeros(1, 1, channelMap.size(3), channelMap.size(4))
    _, max_indices = torch.max(channelMap, 1)
    return max_indices


# local
# batch_maxingChannelMap2compactMap = function(channelMap, n)
# assert (channelMap:dim() == 4)
# local
# compactMaps = torch.zeros(channelMap:size(1), 1, channelMap: size(3), channelMap: size(4))
# for i = 1, channelMap:size(1)
# do
# compactMaps[{{i}, {}, {}, {}}] = maxingChannelMap2compactMap(channelMap[{{i}, {}, {}, {}}], flag)
# end
# return compactMaps
# end

def batch_maxingChannelMap2compactMap(channelMap, n):
    assert channelMap.dim == 4
    compactMaps = torch.zeros(channelMap.size(1), 1, channelMap.size(3), channelMap.size(4))
    for i in (0, channelMap.size(1)):
        compactMaps[i, :, :, :] = maxingChannelMap2compactMap(channelMap[i, :, :, :])
    return compactMaps


# local dispSurrogate = function(t, wid, txt, method)
#     assert(t:dim() == 4)
#     txt = txt or ''
#     if not(method) then
#         disp.image(t, {win=wid, title = tostring(wid) .. ': ' .. txt})
#     elseif method == 'c2cl' then --channel2compactLabels
#         disp.image(batch_channelMap2compactMap(t, 0), {win=wid, title = tostring(wid) .. ': ' .. txt})
#     elseif method == 'c2cc' then --channel2compactCascades
#         disp.image(batch_channelMap2compactMap(t, 1), {win=wid, title = tostring(wid) .. ': ' .. txt})
#     elseif method == 'm2c' then -- maxingChannel2compact
#         disp.image(batch_maxingChannelMap2compactMap(t),{win=wid,title = tostring(wid) .. ': ' .. txt})
#     else
#         error('Not implemented')
#     end
# end
#
# return dispSurrogate


def dispSurrogate(t, wid, txt='', method=''):
    assert t.dim() == 4
    image_title = str(wid) + ': ' + txt
    if not method:
        vis.images(t, win=wid, opts=dict(title=image_title))
    elif method == 'c2cl':
        vis.images(batch_channelMap2compactMap(t, 0), win=wid, opts=dict(title=image_title))
    elif method == 'c2cc':
        vis.images(batch_channelMap2compactMap(t, 1), win=wid, opts=dict(title=image_title))
    elif method == 'm2c':
        vis.images(batch_maxingChannelMap2compactMap(t), win=wid, opts=dict(title=image_title))
    else:
        raise Exception("Not implemented dispSurrogate method!")
