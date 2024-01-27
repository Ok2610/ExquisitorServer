from flask import Flask, request
from flask_cors import CORS
from utils import JSON_MIME_TYPE, make_response, json_response
from pathlib import Path

import sys
import argparse
import string
import datetime
import json
import random
import requests
import datetime
from time import time, sleep, strftime

def print_current_time():
    print(datetime.datetime.now().strftime('%H:%M:%S'))

print_current_time()

import difflib
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
df_close_matches = difflib.get_close_matches

import exq

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--load', action='store_true', default=False)
parser.add_argument('--mod-info-file', type=Path, default='all_mods.json')
parser.add_argument('--common-src-path', type=str, default='', help='If all media paths are the same')
parser.add_argument('--thumb-path', type=str, default='', help='Path to thumbnail directory')
parser.add_argument('--kf-path', type=str, default='', help='Path to keyframes directory')
parser.add_argument('--vid-path', type=str, default='', help='Path to videos directory')

args = parser.parse_args()

idx_cnv = lambda i, load: str(i) if load else i
map_list = lambda m: [(k,m[k]) for k in m]


#########################
# Log files
#########################
ilogname = "log/interactive_log.json"
rlogname = "log/result_log.json"
logname = "log/events" + str(int(time())) + ".json"
with open(logname,'w') as f:
    json.dump([],f)

######################
# GLOBALS & CONSTANTS
######################
TOTAL_IMAGES = 0 # Calculated
TOTAL_VIDS = 0 # Calcualted
N_SUGGS = 64 # Move to config file
N_GRID = 64 # Move to config file
b = 128 # Move to config file
segments = int(b/16)

# TODO: PSQL
# Text file containing all the keyframe filenames of the dataset
ids_file = 'metadata/shotsinfo/full_kf.txt'

ids_map = dict()
ids_reverse_map = dict()

app = Flask(__name__)
CORS(app)

IL = {}
IL['events'] = []

# TODO: PSQL
# Info map from video to shots or other form of grouping
mediainfo_file = 'metadata/shotsinfo/full_mediainfo.json'

with open(mediainfo_file, 'r') as f:
    mediainfo = json.load(f)

# TODO: PSQL
# In case the media items in the dataset is a shot referring to a video, this maps to the timestamp of the shot
kf_ts_file = 'metadata/shotsinfo/full_ts_map.json'
with open(kf_ts_file, 'r') as f:
    timestamp_map = json.load(f)

print('Byte size of timestamp_map:', sys.getsizeof(timestamp_map))

imgVidMap = {}
for key in mediainfo:
    for arr in mediainfo[key]['shots']:
        imgVidMap[arr['exqId']] = key

print('Byte size of imgVidMap:', sys.getsizeof(imgVidMap))

# TODO: PSQL all filters

# Load Filters
place_file = 'metadata/filters/places_t1.json'
place_name_file = 'metadata/filters/places365_classes.txt'
places_names = []
with open(place_name_file,'r') as f:
    places_names = [s.strip() for s in f.readlines()]


# 2D-Array - ExqId -> [Dominant Color, [Color Pallette]]
colors_file = 'metadata/filters/all_keyframe_colors.json'
# Array - Colors
colors_dist = 'metadata/filters/distinct_colors.json'
clrs_name_map = {}
clrs_names = []
with open(colors_dist, 'r') as f:
    clrs_names = json.load(f)
    for i,c in enumerate(clrs_names):
        clrs_name_map[c] = i

# Array - Map of Object Count: Key = ObjectId, Val = Count
objects_file = 'metadata/filters/objectcounts.json'
# Array - Object Name
objects_name_f = 'metadata/filters/detectron_obj_names.json'
obj_names = []
with open(objects_name_f,'r') as f:
    obj_names = json.load(f)


if not args.load:
    with open(ids_file) as f:
        vid_id = 0
        last_vid = 0
        for idx, ss in enumerate(f.readlines()):
            ids_map[idx] = dict()
            ids_map[idx]['name'] = ss.strip()[:-4]
            ids_map[idx]['objects'] = {}
            ids_map[idx]['colorId'] = []
            ids_map[idx]['placeId'] = 365 # unknown
            ids_map[idx]['vidId'] = int(ss.split('_')[0])-1
            ids_reverse_map[ss] = idx
    TOTAL_IMAGES = len(ids_map)

    with open(objects_file, 'r') as f:
        objects = json.load(f)
        for idx,objdict in enumerate(objects):
            ids_map[idx]['objects'] = objdict
            if idx < 4:
                print(objdict)

    #print('ids_map length',len(ids_map))
    with open(colors_file, 'r') as f:
        colors = json.load(f)
        for idx,c in enumerate(sorted(colors.keys())):
            if len(colors[c]) == 0:
                ids_map[idx]['colorId'] = [clrs_name_map['unknown']]
            else:
                ids_map[idx]['colorId'] = [clrs_name_map[colors[c][0]]] + [clrs_name_map[pal] for pal in colors[c][1]]

    with open(place_file, 'r') as f:
        places = json.load(f)
        for idx,p in enumerate(places):
            ids_map[idx]['placeId'] = p
else:
    with open('metadata/saved/ids_map.json','r') as f:
        ids_map = json.load(f)

    with open('metadata/saved/ids_reverse_map.json','r') as f:
        ids_reverse_map = json.load(f)


print('Byte size of ids_map:', sys.getsizeof(ids_map))
print('Loaded ids_map')

# Video level filters
# 2D-Array - VidId -> Categories
vid_category_file = 'metadata/filters/categories.json'
categories_file = open('metadata/filters/distinct_categories.txt', 'r')
category_raw = map(str.strip, categories_file.readlines())
category_descs = []
for c in category_raw:
    category_descs.append(c[12:])
categories_dict = dict()
for idx, s in enumerate(category_descs):
    categories_dict[s] = idx
categories_file.close()
del category_raw

with open(vid_category_file, 'r') as f:
    vid_categories = json.load(f)

categories = []
last_vid = 0
for vid in vid_categories:
    cat_ids = []
    for cat in vid_categories[vid]:
        cat_ids.append(categories_dict[cat[12:]])
    categories.append(cat_ids)
    last_vid += 1

del vid_categories

for rem in range(last_vid,TOTAL_VIDS):
    categories.append([])

#print('cats:',categories[0:3])
print('Byte size of categories:', sys.getsizeof(categories))
print('Byte size of categories_dict:', sys.getsizeof(categories_dict))
print('Loaded categories')

# 2D-Array - VidId -> Tags
vid_tag_file = 'metadata/filters/tags.json'
tags_file = open('metadata/filters/distinct_tags.txt','r')
tags_descs = list(map(str.strip, tags_file.readlines()))
tags_dict = dict()
#tags_map = {}
for idx, s in enumerate(tags_descs):
    tags_dict[s] = idx
#    tags_map[idx] = []
tags_file.close()

with open(vid_tag_file, 'r') as f:
    vid_tags = json.load(f)

last_vid = 0
tags = []
for vid in vid_tags:
    tag_ids = []
    for tag in vid:
        tag_ids.append(tags_dict[tag])
    tags.append(tag_ids)
    last_vid += 1

del vid_tags

print('Byte size of tags:', sys.getsizeof(tags))
print('Byte size of tags_dict:', sys.getsizeof(tags_dict))
print('Loaded tags')


if args.save:
    with open('metadata/saved/ids_map.json','w') as f:
        json.dump(ids_map,f)
    with open('metadata/saved/ids_reverse_map.json','w') as f:
        json.dump(ids_reverse_map,f)

print_current_time()

TOTAL_IMAGES = len(ids_map)
print('Length of total dataset: %d' % TOTAL_IMAGES)
mod_info = []
if not args.mod_info_file.is_file():
    print('Modality information file for Exquisitor not found!')
    raise ValueError
with args.mod_info_file.open('r') as f:
    mod_info = json.load(f)

iota = 1
noms = 1000
num_workers = 1
num_modalities = len(mod_info)
mod_weights = [c['mod_weight'] for c in mod_info]
mod_feature_dimensions = [c['total_feats'] for c in mod_info]
indx_conf_files = [c['indx_path'] for c in mod_info]
func_type = 2 # Different modality weights + mask will be pow(2,mask)-1 in C++
func_objs = []
for m,c in enumerate(mod_info):
    func_objs.append([
        c['n_feat_int']+1,
        c['bit_shift_t'],
        c['bit_shift_ir'],
        c['bit_shift_ir'],
        c['decomp_mask_t'],
        float(pow(2, c['multiplier_t'])),
        c['decomp_mask_ir'],
        c['decomp_mask_ir'],
        float(pow(2,c['multiplier_ir'])),
        c['mod_weight']
    ])
n_items = len(ids_map)
item_metadata = []
vid_ids = set()
vid = -1 # start from -1 since vid += 1 is going to run in the loop to make it 0
for i in range(n_items):
    if mediainfo[imgVidMap[i]]['vidId'] not in vid_ids:
        # print('Adding key', filters[i]['vidId'], ' with value', i)
        vid_ids.add(mediainfo[imgVidMap[i]]['vidId'])
        vid += 1
    item = \
        [0, True, vid, #(int(mediainfo[imgVidMap[i]]['vidId'])-1),
            [], #std_props
            [
                ids_map[idx_cnv(i,args.load)]['colorId'],
                [ids_map[idx_cnv(i,args.load)]['placeId']],
            ], #collection_props
            [
                [[[int(k),int(ids_map[idx_cnv(i,args.load)]['objects'][k])] for k in ids_map[idx_cnv(i,args.load)]['objects']]],
            ] #count_props
        ]
    item_metadata.append(item)
video_metadata = [[]]
print('Number of videos', len(vid_ids))
for i in range(len(vid_ids)):
    vid = [
        categories[i],
        tags[i]
    ]
    video_metadata[0].append(vid)
exp_type = 0
stat_level = 0
try:
    exq.initialize(iota, noms, num_workers, segments, num_modalities, b, indx_conf_files, mod_feature_dimensions,
                func_type, func_objs, item_metadata, video_metadata, exp_type, stat_level, False, 0)
except Exception as e:
    print(e)


test_descs = [0,1,2,3,4,5]
descs = exq.get_descriptors_info(test_descs, 0)
print(descs)

print_current_time()


#####################
# Private Functions
#####################
def convert_item_name_to_video_name(idx, item_name):
    return item_name.split('_')[0]


def event_log(ts, func, session, model, pos=[], neg=[], seen=[], suggs=[], model2='', pos2=[], neg2=[], seen2=[], mergeResIds=[], mergeMethod='', temporalVal='', temporalOption=''):
    log = []
    o = {}
    o['func'] = func
    o['ts'] = ts
    o['session'] = session
    o['model'] = model

    while(1):
        with open(logname, 'r') as f:
            try:
                log = json.load(f)
                break
            except:
                continue

    if func == 'InitModel':
        o['suggs'] = pos
    elif func == 'RandomSet':
        o['suggs'] = pos
    elif func == 'MergeQuery':
        o['pos'] = pos
        o['neg'] = neg
        o['seen'] = seen
        o['model2'] = model2
        o['pos2'] = pos2
        o['neg2'] = neg2
        o['seen2'] = seen2
        o['merge_results'] = mergeResIds
    elif func == 'Learn' or func == 'RandomTheme':
        o['pos'] = pos
        o['neg'] = neg
        o['seen'] = seen
        o['suggs'] = suggs
    elif func == 'ApplyFilters':
        o['filters'] = sessions[session]['models'][model]
    elif func == 'GetImageInfo':
        o['imageId'] = pos[0]
        o['vidId'] = neg[0]
    elif func == 'GetVidInfo':
        o['video'] = pos[0]
    elif func == 'GetSearchItems' or func == 'NextSearchItems' or func == 'PrevSearchItems':
        o['terms'] = pos
    elif func == 'Submit':
        o['submission'] = pos

    log.append(o)
    with open(logname, 'w') as f:
        json.dump(log, f)


def interactive_logging(ts, cat, typ, value):
    global IL
    event = {}
    event['timestamp'] = ts
    event['category'] = cat
    event['type'] = typ
    event['value'] = value
    IL['events'].append(event)


def saveAndResetIL():
    global IL
    with open(ilogname, 'a') as f:
        f.write(json.dumps(IL))
        f.write('\n')

    IL = {}
    IL['events'] = []


def result_logging(ts, cats, types, stype, avail, res, isMap):
    RL = {}
    RL['usedCategories'] = cats
    RL['usedTypes'] = types
    RL['sortType'] = stype
    RL['resultSetAvailability'] = avail
    RL['type'] = 'result'
    RL['ts'] = ts
    if isMap:
        RL['results'] = res
    else:
        RL['results'] = []
        for i in res:
            vid = convert_item_name_to_video_name(i, ids_map[idx_cnv(i,args.load)]['name'])
            shot = ids_map[idx_cnv(i,args.load)]['name'].split('_')[-1]
            entry = {}
            entry['video'] = vid
            entry['shot'] = shot
            RL['results'].append(entry)

    #TODO: Send RL to VBS server

    with open(rlogname, 'a') as f:
        f.write(json.dumps(RL))
        f.write('\n')
    return


def addsuggs (suggs, seen, target):
    suggs_l = len(suggs)
    while suggs_l != target:
        r = random.randrange(0, TOTAL_IMAGES)
        if r not in suggs or r not in seen:
            suggs.append(r)
            suggs_l = len(suggs)

    return suggs


def addModel (u, m):
    model = {}
    model['objects'] = []
    model['colors'] = []
    model['places'] = []
    model['categories'] = []
    model['tags'] = []
    model['exc_list'] = []
    model['cache_items'] = []
    model['cache_pos'] = []
    model['cache_neg'] = []
    model['cache_seen'] = []
    global sessions
    sessions[u]['models'][m] = model
    print(sessions)


def getSuggestions (session, model, nSuggs, pos, neg, seen, qbi=-1):
    print(session)
    print(model)
    print(nSuggs)
    print(pos)
    print(neg)
    print(seen)
    print(qbi)
    print('model', model, sessions[session]['models'][model])
    trainItems = pos + neg
    trainLabels = [1.0 for x in range(len(pos))] + [-1.0 for x in range(len(neg))]
    exq.reset_model(False,False)
    filters = []
    collections = []
    neg_collections = []
    videos = []
    neg_videos = sessions[session]['models'][model]['exc_list']
    std_filters = []
    neg_std_filters = []
    coll_filters = [
            [
                sessions[session]['models'][model]['colors'],
                sessions[session]['models'][model]['places']
            ]
    ]
    neg_coll_filters = []
    vid_filters = [
            [
                sessions[session]['models'][model]['categories'],
                sessions[session]['models'][model]['tags']
            ]
    ]
    neg_vid_filters = []
    objects = []
    # [objid, op, val0, val1]
    for o in sessions[session]['models'][model]['objects']:
        # amount options are [0,1,2,3,4,5,Many]
        op = 0 # EQ
        val0 = o['amount']
        if val0 == 6:
            op = 1 # GTE
        elif val0 == 7:
            op = 0
            val0 = 1
        objects.append([o['object'], op, val0, val0])
    rng_filters = []
    if len(objects) > 0:
        print(objects)
        rng_filters = [
            [
                objects
            ]
        ]
    filters = [collections, videos, std_filters, coll_filters, vid_filters, neg_collections, neg_videos, neg_std_filters, neg_coll_filters, neg_vid_filters, rng_filters]
    train_times = exq.train(trainItems, trainLabels, True, filters, False)
    print('Trained', train_times)
    return exq.suggest(nSuggs, segments, seen, False, [])


# Tokenizer for search string
tokenizer = string.printable

# Global list of items from current search
# search_items = []
# search_start = 0
# search_page_cnt = 54


#######################
# Public API Functions
#######################
sessions = {}

@app.before_request
def before_request():
    print('Exquisitor at your service!')

@app.route("/initExquisitor", methods=['GET'])
def init_exq():
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        print(error)
        return json_response(error, 400)
    global sessions
    session_token = ''
    while (1):
        session_token = ''.join(random.choice(tokenizer) for i in range(20))
        if session_token in sessions:
            continue
        else:
            create = True
            for u in sessions:
                if sessions[u]['ip'] == request.remote_addr:
                    create = False
                    sessions[u]['models'] = {}
                    session_token = u
            if create:
                sessions[session_token] = {}
                sessions[session_token]['models'] = {}
                sessions[session_token]['ip'] = request.remote_addr
            break
    print('sessions',sessions)
    exq.reset_model(False, False)
    ts = int(time())
    event_log(ts, 'InitExquisitor', session_token, {})
    content = json.dumps({ 'session': session_token, 'success': True })
    response = make_response(content, 200, {'Content-Type': 'application/json','Access-Control-Allow-Origin': request.headers['Origin']})
    return response


###################
# Model Functions
###################
@app.route("/initModel", methods=['POST'])
def init_model():
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    print(data)
    grps = data['groups']
    gridgrps = []
    return_total = 0
    for grp in grps:
        return_total += grp['itemsToShow']
    
    r_ids = random.sample(sorted(ids_map.keys()), return_total)
    pre_last_index = 0
    for grp in grps:
        idx_start = pre_last_index
        idx_end = pre_last_index + grp['itemsToShow']
        gridgrps.append({
            'id': data['modelId'],
            'itemsToShow': grp['itemsToShow'],
            'items': r_ids[idx_start:idx_end]
        })
    
    session = data['session']
    model = data['modelId']
    addModel(session,model)
    ts = int(time())
    exq.reset_model(False, False)

    result_logging(ts, ['Browsing'], ['randomSelection'], '', 'sample', r_ids, False)
    event_log(ts, 'InitModel', session, model, r_ids)

    content = json.dumps({ 'groups': gridgrps })
    response = make_response(content, 200, {'Content-Type': 'application/json','Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/getTotalImages", methods=['GET'])
def init_get_total_images():
    content = json.dumps({'total_images': TOTAL_IMAGES})
    response = make_response(content, 200, {'Content-Type': 'application/json','Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/resetModel", methods=['GET'])
def reset_model():
    exq.reset_model(False,False)
    data = request.json
    session = data['session']
    model = json.loads(str(data['model']))
    addModel(session,model)
    ts = int(time())

    event_log(ts, 'ResetModel', session, model)

    content = json.dumps({'reset':'successful'})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/deleteModel", methods=['POST'])
def delete_model():
    exq.reset_model(False,False)
    data = request.json
    session = data['session']
    model = json.loads(str(data['model']))
    global sessions
    del sessions[session]['models'][m]
    ts = int(time())

    event_log(ts, 'RemoveModel', session, model)

    content = json.dumps({'reset':'successful'})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/randomSet", methods=['POST'])
def random_set():
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    r_ids = json.loads(str(data['ids']))
    session = data['session']
    model = json.loads(str(data['model']))
    img_locations = [ids_map[idx_cnv(s,args.load)]['name'] for s in r_ids]
    content = json.dumps({'img_locations': img_locations})
    ts = int(time())

    event_log(ts, 'RandomSet', session, model, r_ids)
    interactive_logging(ts, 'Browsing', 'randomSelection','')
    result_logging(ts, ['Browsing'], ['randomSelection'], '', 'sample', r_ids, False)

    response = make_response(content, 200, {'Content-Type': 'application/json','Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/urf", methods=['POST'])
def suggest():
    """
    Performs the User Relevance Feedback process based on the positive and negative items provided.
    It takes an argument 'n' as the number of suggestions to return.
    JSON body: session, model, n, pos, neg
    """
    start = time()
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    print(request)

    session = data['session']
    model = json.loads(str(data['model']))
    nSuggs = json.loads(str(data['n']))
    pos_list = json.loads(str(data['pos']))
    neg_list = json.loads(str(data['neg']))
    seen_list = []
    try:
        seen_list = json.loads(str(data['seen']))
    except:
        print('No seen list provided')
    
    if len(pos_list) == 0:
        pos_list = random.sample(range(0,TOTAL_IMAGES),1)

    if len(neg_list) == 0:
        neg_list = random.sample(range(0,TOTAL_IMAGES),1)
    
    # exc_list = json.loads(str(data['excludedVids']))
    # queryByImage = json.loads(str(data['queryByImage']))
    global sessions
    # sessions[session]['models'][model]['exc_list'] = exc_list
    t_test = int(time())
    (sugg_list, total, worker_time, sugg_time, sugg_overhead) = getSuggestions(session, model, nSuggs, pos_list, neg_list, seen_list)
    print(total, worker_time, sugg_time, sugg_overhead)
    t = int(time()) - t_test

    ts = int(time())

    event_log(ts, 'Learn', session, model, pos_list, neg_list, seen_list, sugg_list)
    interactive_logging(ts, 'Image', 'feedbackModel', [[ids_map[idx_cnv(i,args.load)]['name'] for i in pos_list],[ids_map[idx_cnv(i,args.load)]['name'] for i in neg_list],[ids_map[idx_cnv(i,args.load)]['name'] for i in seen_list]])

    resMap = []
    if (len(sugg_list) > 0):
        for sug in sugg_list:
            entry = {}
            vid = convert_item_name_to_video_name(int(sug), ids_map[idx_cnv(int(sug),args.load)]['name'])
            shot = ids_map[idx_cnv(int(sug),args.load)]['name'].split('_')[-1]
            entry['video'] = vid
            entry['shot'] = shot
            resMap.append(entry)

        result_logging(ts, ['Image'], ['feedbackModel'], '', 'top', resMap, True)

    # img_locations = [ids_map[idx_cnv(s,args.load)]['name'] for s in sugg_list]

    print(sugg_list)
    # TODO: Update response to match new JSON Item object
    content = json.dumps({'suggestions': sugg_list})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    #experiment 2
    end = int(time())
    print("Learn with k: " + str(nSuggs) + " time: " + str(end-start))

    return response


#############################
# Getters for Items + Info
#############################
@app.route("/getCacheItem", methods=['GET'])
def getCacheItem():
    # TODO: Get more than requested suggest items
    # TODO: If model hasn't changed get from the cache otherwise call suggest()
    content = json.dumps({})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/getItem", methods=['POST'])
def get_item():
    """
    Given an item id, this function returns its media source information.
    """
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    item_id = json.loads(str(data['itemId']))
    kf_name = ids_map[idx_cnv(item_id,args.load)]['name']
    vid_id = ids_map[idx_cnv(item_id,args.load)]['vidId']
    thumb_path = ''
    src_path = ''
    media_type = 0
    if args.common_src_path != '':
        thumb_path = '/'.join('/', [args.common_src_path, 'thumbnails', kf_name + '.jpg'])
        src_path = '/'.join([args.common_src_path, 'keyframes', str(vid_id+1).zfill(5), kf_name + '.jpg'])
    elif args.thumb_path != '' or args.kf_path != '' or args.vid_path != '':
        if args.thumb_path != '':
                thumb_path = '/'.join([args.thumb_path, kf_name + '.jpg'])
        if args.kf_path != '':
                src_path = '/'.join([args.kf_path, kf_name + '.jpg'])
        if args.vid_path != '':
                src_path = '/'.join([args.vid_path, kf_name + '.mp4'])
                media_type = 1
    else:
        thumb_path = kf_name + '.jpg'
        src_path = kf_name + '.jpg'
    
    vid = convert_item_name_to_video_name(item_id, kf_name)
    related_items = {
        'timelineN': len(mediainfo[vid]['shots']),
        'timelineRange': [mediainfo[vid]['shots'][0]['exqId'], mediainfo[vid]['shots'][-1]['exqId']]
    }

    content = json.dumps({ 
        'id': item_id,
        'name': kf_name,
        'mediaId': item_id,
        'mediaType': media_type,
        'thumbPath': thumb_path,
        'srcPath': src_path,
        'relatedItems': related_items
    })
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/getItemInfo", methods=['POST'])
def get_item_info():
    """
    Returns the metadata information of a given item.
    JSON body: session, model, itemId
    """
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    session = data['session']
    model = json.loads(str(data['model']))
    item_id = json.loads(str(data['itemId']))
    vidId = ids_map[idx_cnv(item_id,args.load)]['vidId']

    if (item_id > TOTAL_IMAGES):
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    ts = int(time())

    event_log(ts, 'GetItemInfo', session, model, [item_id], [vidId])
    interactive_logging(ts, 'Browsing', 'toolLayout', item_id)
    interactive_logging(ts, 'Browsing', 'videoSummary', item_id)

    name = ids_map[idx_cnv(item_id,args.load)]['name']
    objects = [(obj_names[int(k)],v) for k,v in map_list(ids_map[idx_cnv(item_id,args.load)]['objects'])]
    colors = [clrs_names[c] for c in ids_map[idx_cnv(item_id,args.load)]['colorId']]
    places = places_names[ids_map[idx_cnv(item_id,args.load)]['placeId']]
    ctgs = [category_descs[id] for id in categories[vidId]]
    tgs = [tags_descs[id] for id in tags[vidId]]

    # Return ItemInfo { infoPair: [string, string[]][] }
    content = json.dumps({'infoPair': [['ID', [item_id]], ['Name', [name]], ['Categories', ctgs], ['Tags', tgs], ['Objects', objects], ['Places', [places]], ['Colors', colors]]})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/getRelatedItems", methods=['POST'])
def get_related_items():
    """
    Gets the related shots/items for a given item.
    JSON body: itemId
    """
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    item_id = json.loads(str(data['itemId']))
    vid = convert_item_name_to_video_name(item_id, ids_map[idx_cnv(item_id,args.load)]['name'])

    total_shots = len(mediainfo[vid]['shots'])
    start = mediainfo[vid]['shots'][i]['exqId']
    end = mediainfo[vid]['shots'][-1]['exqId']

    # Return RelatedItems { timelineN: number, timelineRange: [number,number]}
    content = json.dumps({'timelineN': total_shots, 'timelineRange': [start,end]})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/getVidInfo", methods=['POST'])
def get_media_info():
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    session = data['session']
    model = json.loads(str(data['model']))
    video = data['video']
    vidId = int(video)
    # itemId = vidmap[video][0][0]
    itemId = mediainfo[video]['shots'][0]['exqId']
    numShots = json.loads(str(data['numShots']))
    ts = int(time())

    event_log(ts, 'GetVidInfo', session, model, [video])
    interactive_logging(ts, 'Browsing', 'toolLayout', video)
    interactive_logging(ts, 'Browsing', 'videoSummary', video)

    name = ids_map[idx_cnv(itemId,args.load)]['name']
    objects = [(obj_names[int(k)],v) for k,v in map_list(ids_map[idx_cnv(itemId,args.load)]['objects'])]
    colors = [clrs_names[c] for c in ids_map[idx_cnv(itemId,args.load)]['colorId']]
    places = places_names[ids_map[idx_cnv(itemId,args.load)]['placeId']]
    ctgs = [category_descs[id] for id in categories[vidId]]
    tgs = [tags_descs[id] for id in tags[vidId]]
    exqIds = []
    shotFrames = []
    start = 0
    end = numShots

    for i in range(start,end):
        kf = mediainfo[video]['shots'][i]['thumbnail']
        # samples = mediainfo[video]['shots'][i]['samples']
        exqIds.append(mediainfo[video]['shots'][i]['exqId'])
        # shotFrames.append(getShotFrames(kf,samples))

    location = ids_map[idx_cnv(itemId,args.load)]['name']
    content = json.dumps({'id': itemId, 'itemLocation': location, 'name': name, 'categories': ctgs, 'tags':tgs, 'objects': objects,
                           'places': places, 'colors': colors, 'shots': shotFrames, 'exqIds': exqIds})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


##########
# Search
##########

@app.route("/searchVLM", methods=['POST'])
def searchVLM():
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    session = data['session']
    model = json.loads(str(data['model']))
    query = data['query']
    positives = json.loads(str(data['positives']))

    vlm_query = {'query': query}
    vlm_results = requests.post('localhost:5010/retrieveImages', json=vlm_query)
    vlm_rewrite = {'query': query, 'positives': positives}
    vlm_caption = requests.post('localhost:5010/rewriteQuery', json=vlm_rewrite)

    ts = int(time())
    event_log(ts, 'SearchVLM', session, model, [query], positives)
    interactive_logging(ts, 'Image/Text', 'conversationSearch', (query,positives))
    result_logging(ts, ['Text'], ['conversationSearch'], '', 'top', vlm_results, False)

    content = json.dumps({'suggestion': vlm_caption, 'top': vlm_results})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response


############
# Filters
############
@app.route("/getFilters", methods=['GET'])
def get_filters():
    """
    Sends the list of filters a user can apply on the current collection(s)
    """
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    filters = [
        {
            'id': 0,
            'collectionId': 'C1',
            'name': 'Categories',
            'values': category_descs,
            'filter': 1, # Multi
        },
        {
            'id': 1,
            'collectionId': 'C1',
            'name': 'Tags',
            'values': tags_descs,
            'filter': 1, # Multi
        },
        {
            'id': 2,
            'collectionId': 'C1',
            'name': 'Places',
            'values': places_names,
            'filter': 1, # Multi
        },
        {
            'id': 3,
            'collectionId': 'C1',
            'name': 'Dominant Color',
            'values': clrs_names,
            'filter': 1, # Multi
        },
        # TODO: Set correct options for Count filter
        # {
        #     'id': 4,
        #     'collectionId': 'C1',
        #     'name': 'Objects',
        #     'values': obj_names,
        #     'filter': 1, # Multi
        #     'range': [],
        #     'rangeLabel': [],
        #     'count': [],
        #     'property': -1
        # }
        # {
        #     'id': 1,
        #     'collectionId': 'C1',
        #     'name': 'Tags',
        #     'values': tags_descs,
        #     'filter': 1, # Multi
        #     'range': [],
        #     'rangeLabel': [],
        #     'count': [],
        #     'property': -1
        # }
    ]
    content = json.dumps({ 'filters': filters })
    response = make_response(content, 200, {'Content-Type': 'application/json','Access-Control-Allow-Origin': request.headers['Origin']})
    return response


@app.route("/applyFilters", methods=['POST'])
def apply_filters():
    """
    Applies the filters for the given session and model id.
    The filters are sent as two lists, the first being the name(s) and second their set value(s).
    JSON body: session, model, names, values
    """
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    print(request)
    ts = int(time())
    session = data['session']
    model = json.loads(str(data['model']))
    filter_names = data['names']
    filter_values = json.loads(str(data['values']))
    global sessions
    # objects = str(data['objects']).replace("\'","\"")
    for idx,filter in enumerate(filter_names):
        sessions[session]['models'][model][filter] = filter_values[idx]
    print(sessions[session]['models'][model])
    event_log(ts, 'ApplyFilters', session, model)
    interactive_logging(ts, 'Filters', 
                        'appliedFilters', (#sessions[session]['models'][model]['objects'], 
                                            sessions[session]['models'][model]['colors'], sessions[session]['models'][model]['places'], sessions[session]['models'][model]['categories'], sessions[session]['models'][model]['tags']))
    content = json.dumps({'log':'successful'})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response

@app.route("/resetFilters", methods=['POST'])
def reset_filters():
    """
    Resets the applied filters for the given session and model id
    JSON boday: session, model
    """
    data = request.json
    session = data['session']
    model = json.loads(str(data['model']))
    addModel(session, model)
    ts = int(time())
    event_log(ts, 'ResetFilters', session, model)
    content = json.dumps({'reset':'successful'})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response

##########
# Logging
##########
@app.route("/logInteraction", methods=['POST'])
def logInteraction():
    if request.content_type != JSON_MIME_TYPE:
        print(request.content_type)
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)
    data = request.json
    ts = data['ts']
    cat = data['category']
    typ = data['type']
    val = data['value']
    interactive_logging(ts, cat, typ, val)
    content = json.dumps({'log':'success'})
    response = make_response(content, 200, {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': request.headers['Origin']})
    return response

if (__name__ == "__main__"):
    app.run(debug=True, use_reloader=False, port = 5001, host = "0.0.0.0")
