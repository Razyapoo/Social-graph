import networkx
import numpy
import requests
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
import time

def full_database(name):
    database = set() #databaze clenu pro jednu skupinu
    mem_count = -1
    offset = 0 #offset nactenych clenu skupiny
    while mem_count != len(database): # je moznost, ze bude nekonecny cyklus
        response = get_users(name, offset=offset)['response']
        if mem_count != response['count']:
            mem_count = response['count'] #pocet clenu ve skupine celkem
        temp_mem = response['items'] # nove cleny
        offset += len(temp_mem) #pocet zpracovanych clenu
        if database | set(temp_mem) == database != set(): # bez novych clenu
            print('BREAK! Infitity loop', mem_count, len(database))
            break
        database = database.union(temp_mem)

    return database

def get_users(group_name, offset=0):
    # http://vk.com/dev/groups.getMembers
    host = 'https://api.vk.com/method' #https://api.vk.com/method/users.get?user_id=210700286&v=5.52 // https://vk.com/dev
    access_token = "b24d1314b24d1314b24d131412b224f6c5bb24db24d1314eebdc406a3c0cb01ab533963"
    version = 5.29
    mem_count = 1000

    if mem_count > 1000:
        raise Exception('max = 1000')

    response = requests.get('{host}/groups.getMembers?group_id={group_id}&offset={offset}&count={count}&access_token={access_token}&v={v}'
                            .format(host=host, group_id=group_name, count=mem_count, offset=offset, access_token=access_token, v=version))

    if not response.ok:
        raise Exception('Error')
    return response.json()

# def pagerank(G, alpha=0.85,  iterations=100, tol=1.0e-6, weight='weight'): 
   
#     if len(G) == 0: 
#         return {} 
  
#     # vytvorime orientovany graf 
#     #D = G.to_directed() 
  
#     # pridame vahy 
#     # udelame stohastickou matice
#     W = networkx.stochastic_graph(G, weight=weight) 
#     N = W.number_of_nodes() 
  
#     # Choose fixed starting vector 
#     v = dict.fromkeys(W, 1.0 / N) 

#     # dangling 
#     # pomoci personalization vector  
#     dangling_weights =  dict.fromkeys(W, 1.0 / N)
    
#     dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0] 
  
    
#     for i in range(iterations): 
#         vlast = v # kopie pole, je potreba pro ulozeni novych hodnot
#         v = dict.fromkeys(vlast.keys(), 0) 
#         danglesum = alpha * sum(vlast[n] for n in dangling_nodes) 
#         for n in v: 
#             for neighbor in W[n]: 
#                 v[neighbor] += alpha * vlast[n] * W[n][neighbor][weight] 
#             v[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * dangling_weights[n] 
  
#         # overeni convergence, l1 norm 
#         err = sum([abs(v[n] - vlast[n]) for n in v]) 
#         if err < N*tol: 
#             return v 
#     raise networkx.NetworkXError('pagerank: power iteration failed to converge '
#                         'in %d iterations.' % iterations) 

def google_matrix(G, alpha=0.85, nodelist=None):

    M=networkx.to_numpy_matrix(G,nodelist=nodelist)
    (n,m)=M.shape
    # dangling uzly
    dangling=numpy.where(M.sum(axis=1)==0)
    for d in dangling[0]:
        M[d]=1.0/n
    # normalizace      
    M=M/M.sum(axis=1)

    P=alpha*M+(1-alpha)*numpy.ones((n,n))/n
    return P

def pagerank_numpy(G,alpha=0.85,max_iter=100,tol=1.0e-6,nodelist=None):
    """Return a NumPy array of the PageRank of G.
    """
    M=google_matrix(G,alpha,nodelist)   
    (n,m)=M.shape 
    x=numpy.ones((n))/n
    for i in range(max_iter):
        xlast=x
        x=numpy.dot(x,M)
        # kontrola konvergence (sblizeni), l1 norm            
        err=numpy.abs(x-xlast).sum()
        if err < n*tol: # error tolerance se pouziva pro konvergence v power metode
            return numpy.asarray(x).flatten()

    raise networkx.NetworkXError("pagerank: power iteration failed to converge in %d iterations."%(i+1))

groups = [
'http://vk.com/luxurydeephouse',
'http://vk.com/good.music',
'http://vk.com/go_deephouse',
'http://vk.com/mp3em',
'http://vk.com/vsemuzlo',
'http://vk.com/deephousemzk',
'http://vk.com/russian_deep_house',
'http://vk.com/deephouseband',
'http://vk.com/fdjs2013',
'http://vk.com/deephouseliveru',
'http://vk.com/deephousewave',
'http://vk.com/sunlessmusic',
'http://vk.com/public_deephouse',
'http://vk.com/deep_house_deep'
'http://vk.com/deephere',
'http://vk.com/monobeat_electronic_music',
'http://vk.com/housecloud',
'http://vk.com/dhbest',
'http://vk.com/edm13',
'http://vk.com/kacheli26',
'http://vk.com/newdeephouse',
'http://vk.com/deep_house_people',
'http://vk.com/deepbaikal',
'http://vk.com/house_department',
'http://vk.com/deep_house_exclusive_community',
'http://vk.com/deep_house_music1',
'http://vk.com/relaxingdp',
'http://vk.com/concept_sound',
'http://vk.com/rudeepradio',
'http://vk.com/deep_houses',
'http://vk.com/best_deephouse',
'http://vk.com/techberry',
'http://vk.com/territory_deep_house',
'http://vk.com/libravo',
'http://vk.com/deep__music.house',
'http://vk.com/deephouse_luxe',
'http://vk.com/deep_house_attack',
'http://vk.com/deep_house_music_group',
'http://vk.com/dhmtt',
'http://vk.com/fashionrhythm',
'http://vk.com/deep_house_collection',
'http://vk.com/musicofdeephouse',
'http://vk.com/cafe_deep_house',
'http://vk.com/music.sunshine',
'http://vk.com/electrodances',
'http://vk.com/gudinisounds',
'http://vk.com/you_club_music',
'http://vk.com/electrotechdeephouse',
'http://vk.com/dj_clarion_official_page',
'http://vk.com/soundeephouse',
'http://vk.com/best_deep_house',
'http://vk.com/house.deep',
'http://vk.com/ibiza_vk',
'http://vk.com/deephouse_1',
'http://vk.com/instagram_moda',
'http://vk.com/winedeep',
'http://vk.com/deep.realtones',
'http://vk.com/umbrellafm',
'http://vk.com/deep_kan',
'http://vk.com/beautiful_body_deep',
'http://vk.com/fomichevdj',
'http://vk.com/galleryofdeephouse',
'http://vk.com/bestofhousemusick',
'http://vk.com/life_is_deep',
'http://vk.com/aviodeep',
'http://vk.com/rudeepradio2',
'http://vk.com/deep.house.russia',
'http://vk.com/music_cars_girls',
'http://vk.com/deephousestation2019',
'http://vk.com/gtokmusic',
'http://vk.com/vk_deep',
'http://vk.com/deephouseclassic',
'http://vk.com/deep1radio',
'http://vk.com/housetimegroup',
'http://vk.com/lunosound',
'http://vk.com/d_e_e_p_h_o_u_s_e',
'http://vk.com/deephouse_new',
'http://vk.com/d_u_b__5_t_e_p'
]

mems = {} #pole uzivatelu v kazde skupine
for g in groups:
    label = g.split('http://vk.com/')[1] #nazev skupiny
    print(label)
    mems[label] = full_database(label) #plny pocet clenu kazde skupiny
    
matrix = {} # matice "sousedu"
mem = {} # spolecne clenove
for i in mems:
    for j in mems:
        if i != j:
            mem = mems[i] & mems[j]
            matrix[i+j] = len(mem) * 1.0 / min(len(mems[i]), len(mems[j]))

max_matrix = max(matrix.values())
min_matrix = min(matrix.values())

for i in matrix:
    matrix[i] = (matrix[i] - min_matrix) / (max_matrix - min_matrix) #normalizace matici
    
g = networkx.DiGraph()
for i in mems:
    g.add_node(i)
    for j in mems:
        if (i != j) and (len(mems[i]) < len(mems[j])) and (len(mems[i] & mems[j]) > len(mems[i]) * 0.1):
            g.add_edge(i, j, weight=matrix[i+j])
            
count = {v: len(mems[v]) for v in mems} #pole poctu clenu skupin

max_val = max(count.values()) * 1.0 # max z poctu clenu zkupin
size_of_nodes = [] #velikosti uzlu
max_size_node = 120
min_size_node = 10
for node in g.nodes():
    size_of_nodes.append(((count[node]/max_val)*max_size_node + min_size_node)*10)#velikost uzlu, na zaklade mnozstvi clenu

positions = networkx.spring_layout(g) #pozice uzlu
plt.figure(figsize=(20, 20))
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
networkx.draw_networkx(g, positions, node_size=size_of_nodes, width=0.5, font_size=8) # graf
plt.axis('off')
#pr = pagerank(g)
num_pr = pagerank_numpy(g)
#google_pr = google_matrix(g)
print(num_pr)
#print(google_pr)
pr_net = networkx.pagerank(g) #pagerank
print(pr_net)
plt.show()






