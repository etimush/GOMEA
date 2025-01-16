import PIL.Image
import numpy as np
import cv2
from numba import jit, njit
#from joblib import Parallel
from scipy.spatial import KDTree
from scipy.spatial import Voronoi
from scipy.stats import chi2_contingency
import time
import cProfile
from PIL import Image, ImageDraw
import csv
import  copy
import multiprocessing as mp
import random
import torchvision.models as models
from torchvision import transforms
import torch
from Evaluators import calcEntropy, neural_net, cv2_norm
import scipy.misc

#Improvement to code ## Parameter genome --> Store parameters such as mutation rate and strength in its own genome
# and mutate tem aswell, if mutation rate is better tha nsome decaying function keep
from networkx.algorithms.components.connected import connected_components
from Lincage_tree import to_graph

class Individual(object):
    def __init__(self, num_points, height, width,ground_t):
        self.place_genotype = self.make_p_genotype(num_points, height, width)
        self.color_genotype = np.random.choice(range(256), p=np.ones(256) / 256, size=3 * num_points)
        self.places_loc = np.zeros(len(self.place_genotype))
        self.color_loc = np.zeros(len(self.color_genotype))
        self.score = np.inf
        self.height = height
        self.width = width
        self.ground_t = ground_t
        self.painting = None



    def make_p_genotype(self, num_points, height, width):
        x = np.random.choice(range(height + 1), p=np.ones(height + 1) / (height + 1), size=num_points)
        y = np.random.choice(range(width + 1), p=np.ones(width + 1) / (width + 1), size=num_points)
        place_g = np.append(x, y, axis=np.newaxis)
        return place_g

    def mutate_genotypes(self, p=0.0075):
        max_xy = self.height -1
        max_c = 256
        mutations = ['shift', 'color']
        weights = [2/4,2/4]
        strenth = np.random.random()*2
        choice = np.random.choice(mutations,p=weights)

        #mutation = np.random.normal(self.places_loc,np.ones(len(self.place_genotype)),len(self.place_genotype))

        if choice == 'shift':
            mask_p = np.random.choice([True, False], size=len(self.place_genotype), p=[p, 1 - p])
            mutation_p = np.random.uniform(low=-(strenth/2), high=(strenth/2),size=len(self.place_genotype))
            #mutation_p = np.random.normal(self.places_loc, np.ones(len(self.place_genotype)), len(self.place_genotype))
            self.places_loc = (self.places_loc + mutation_p)
            mutation_p = (mutation_p*max_xy) + self.place_genotype
            mutation_p = np.where(mutation_p< 0, self.place_genotype, mutation_p)
            mutation_p = np.where(mutation_p > max_xy , self.place_genotype, mutation_p)
            self.place_genotype = np.where(mask_p,mutation_p.astype(int),self.place_genotype)

        if choice == 'color' :
            mask_c = np.random.choice([True, False], size=len(self.color_genotype), p=[p, 1 - p])
            mutation_c = np.random.uniform(low=-(strenth / 2), high=(strenth / 2), size=len(self.color_genotype))
            #mutation_c = np.random.normal(self.color_loc, np.ones(len(self.color_genotype)), len(self.color_genotype))
            self.color_loc = (self.color_loc + mutation_c)
            mutation_c = (mutation_c*max_c) + self.color_genotype
            mutation_c = np.where(mutation_c < 0, self.color_genotype, mutation_c)
            mutation_c = np.where(mutation_c > 255, self.color_genotype, mutation_c)
            self.color_genotype = np.where(mask_c,mutation_c.astype(np.uint8),self.color_genotype)




    def get_params(self):
        return self.place_genotype.reshape((-1, 2)), self.color_genotype.reshape((-1, 3))

    def get_params_loc(self):
        return self.places_loc.reshape((-1, 2)), self.color_loc.reshape((-1, 3))


    def duplicate_genome(self):

        self.color_genotype = np.append(self.color_genotype,copy.deepcopy(self.color_genotype))
        self.place_genotype = np.append(self.place_genotype , copy.deepcopy(self.place_genotype))

    def evaluate_individual(self,):
        veroni_points, veroni_colors = self.get_params()
        self.draw_voronoi_matrix2(veroni_points, veroni_colors, self.width, self.height)
        #paint = cv2.cvtColor(self.painting, cv2.COLOR_RGB2GRAY)
        self.score = cv2.norm(self.ground_t, self.painting, cv2.NORM_L2)   #self.score = fractal_dimension(paint/256)

    def draw_voronoi_matrix2(self,genotype_coords, genotype_colors, img_width, img_height, pre_features=False,
                             pre_features_len=1000):
        coords = []
        for coord in genotype_coords:
            coords.append((coord[0], coord[1]))
        vor = Voronoi(coords)

        canvas = np.ones((img_height, img_width, 3), dtype=np.uint8)
        for i, (point, region_idx) in enumerate(zip(genotype_coords, vor.point_region)):
            polygon = []
            draw = True
            color = (int(genotype_colors[i, 0]), int(genotype_colors[i, 1]), int(genotype_colors[i, 2]))
            for vertex_idx in vor.regions[region_idx]:
                if vertex_idx == -1:
                    draw = False
                i, j = vor.vertices[vertex_idx]
                polygon.append([int(i), int(j)])
            if draw:
                if pre_features:
                    if len(genotype_coords) < pre_features_len:
                        cv2.polylines(img=canvas, pts=np.array([polygon]), color=color, isClosed=True)
                    else:
                        cv2.fillPoly(img=canvas, pts=np.array([polygon]), color=color)
                else:
                    cv2.fillPoly(img=canvas, pts=np.array([polygon]), color=color)

        self.painting = canvas



    def removal_event(self):
        num = range(int((len(self.place_genotype)/2) -1))

        places = np.random.choice(num,size=int(0.1*len(self.place_genotype)), replace= False)

        p1 = lambda x: 2*x
        p2 = lambda x: (2*x) +1

        c1 = lambda x: 3*x
        c2 = lambda x: (3 * x) + 1
        c3 = lambda x: (3 * x) + 2
        p_i = [ps(place) for place in places for ps in (p1,p2)]
        c_i = [ps(place) for place in places for ps in (c1,c2,c3)]

        self.place_genotype = np.delete(self.place_genotype,p_i)
        self.color_genotype = np.delete(self.color_genotype, c_i)

@njit()
def norm(g_t,image):
    score = 0
    for i in range(g_t.shape[0]):
        for j in range(g_t.shape[1]):
            score += np.sum(np.abs(g_t[i,j,:]-image[i,j,:]))
    return score

def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return coeffs[0]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def evaluate_individuals(individuals, ground_t,width,height):
    for ind in individuals:
        ind.evaluate_individual()



def evaluate_imgs( ground_t,painting):
    #im = Image.fromarray(painting)
    #ground_truth = Image.fromarray(ground_t)
    return cv2.norm( ground_t, painting, cv2.NORM_L2 )#image_diff(ground_truth, im)

def tournament_selection(population):
    selection_pool = copy.deepcopy(population)
    tournament_size = 2
    selection = []
    assert len(selection_pool) % tournament_size == 0, "Population size should be a multiple of tournament size"
    for _ in range(int(tournament_size / (len(selection_pool) / len(population)))):
        np.random.shuffle(selection_pool)
        for i in range(0, len(selection_pool), tournament_size):
            candidates = [selection_pool[i + j] for j in range(tournament_size)]
            candidates.sort(key=lambda individual: individual.score)
            selection.append(copy.deepcopy(candidates[0]))

    assert len(population) == len(selection), "Selection size should be a multiple of tournament size"

    return selection


def draw_voronoi_matrix(genotype_coords, genotype_colors, img_width, img_height):
    coords = []
    for coord in genotype_coords:
        coords.append((coord[0],coord[1]))
    colors = genotype_colors
    voronoi_kdtree = KDTree(coords)
    query_points = [(x, y) for x in range(img_width) for y in range(img_height)]

    _, query_point_regions = voronoi_kdtree.query(query_points)

    data = np.zeros((img_height, img_width, 3), dtype='uint8')
    i = 0
    for x in range(img_height):
        for y in range(img_width):
            data[y, x, :] = colors[query_point_regions[i]]
            i += 1

    return data




def duplication_event(individuals):
    for ind in individuals:
        ind.duplicate_genome()

def removal_event(individuals):
    for ind in individuals:
        ind.removal_event()


def linkage_tree(individuals):
    l_t = []

    places, colors = individuals[0].get_params()
    tiled = np.tile(places, (places.shape[0], 1, 1))
    tiled_t = np.transpose(tiled, axes=(1, 0, 2))
    tiled_c = np.tile(places, (colors.shape[0], 1, 1))
    tiled_t_c = np.transpose(tiled_c, axes=(1, 0, 2))

    dists = np.linalg.norm(tiled_t - tiled,ord=2, axis=2)
    dists += np.linalg.norm(tiled_t_c - tiled_c,ord=2, axis=2)
    dists[dists == 0] = np.inf

    for i in range(1,len(individuals)):
        places,colors = individuals[i].get_params()
        tiled = np.tile(places, (places.shape[0], 1, 1))
        tiled_t= np.transpose(tiled, axes=(1,0,2))
        tiled_c = np.tile(places, (colors.shape[0], 1, 1))
        tiled_t_c = np.transpose(tiled_c, axes=(1, 0, 2))
        distances = np.linalg.norm(tiled_t - tiled,ord=2 ,axis=2)
        distances[distances == 0] = np.inf
        dists = dists + distances + np.linalg.norm(tiled_t_c - tiled_c,2, axis=2)
    dists = dists/len(individuals)


    for i in range(int(dists.shape[0])):
        l_t.append([i,np.argmin(dists[i])])

    #G = to_graph(l_t)
    #l_t =  [list(c) for c in connected_components(G)]
    #print("Linkage Tree length: ", len(l_t))

    return l_t


def gene_optimal_mixing(individuals,l_t,ground_t,width, height,x,i = 0,asynch = False):
    o_i = copy.deepcopy(x)
    o_i_places, o_i_colors = o_i.get_params()
    o_i_places_loc, o_i_colors_loc = o_i.get_params_loc()
    best  =  copy.deepcopy(x)
    best_places, best_colors = best.get_params()
    best_places_loc, best_colors_loc = best.get_params_loc()
    parents = copy.deepcopy(individuals)
    mutate_individuals(parents, p=0.75)


    for subset in l_t:
        subset = subset[0]
        p = np.random.choice(parents)
        p_places, p_colors = p.get_params()
        p_places_loc, p_colors_loc = p.get_params_loc()
        o_i_places[subset] = p_places[subset]
        o_i_colors[subset] = p_colors[subset]

        o_i_places_loc[subset] = p_places_loc[subset]
        o_i_colors_loc[subset] = p_colors_loc[subset]

        if not np.all(o_i_places[subset] == best_places[subset]) or not np.all(o_i_colors[subset] == best_colors[subset]):
            o_i.places_genotype = o_i_places.ravel()
            o_i.colors_genotype = o_i_colors.ravel()

            o_i.places_loc = o_i_places_loc.ravel()
            o_i.color_loc = o_i_colors_loc.ravel()

            o_i.evaluate_individual()
            if o_i.score < best.score:
                best_places[subset] = o_i_places[subset]
                best_colors[subset] = o_i_colors[subset]

                best_places_loc[subset] = o_i_places_loc[subset]
                best_colors_loc[subset] = o_i_colors_loc[subset]



                best.places_genotype = best_places.ravel()
                best.colors_genotype = best_colors.ravel()

                best.places_loc = best_places_loc.ravel()
                best.color_loc = best_colors_loc.ravel()

                best.score = o_i.score

            else:

                o_i_places[subset]= best_places[subset]
                o_i_colors[subset] = best_colors[subset]

                o_i_places_loc[subset] = best_places_loc[subset]
                o_i_colors_loc[subset] = best_colors_loc[subset]

                o_i.places_genotype = o_i_places.ravel()
                o_i.colors_genotype = o_i_colors.ravel()

                o_i.places_loc = o_i_places_loc.ravel()
                o_i.color_loc = o_i_colors_loc.ravel()

                o_i.score = best.score

    if not asynch:
        return o_i
    else: return (i,o_i)

def variation(individuals,l_t,ground_t,width,height,type='parralel_GOM'):
    offspring = []

    if type == 'GOM':
        for individual in individuals:
            offspring.append(gene_optimal_mixing(individuals, l_t, ground_t, width, height, individual))
        individuals = offspring

    if type == "parralel_GOM":
        pool = mp.Pool(mp.cpu_count())
        offspring = pool.starmap_async(gene_optimal_mixing, [( individuals,l_t, ground_t, width, height, individual, i , True) \
                                                   for i, individual in enumerate(individuals)]).get()
        pool.close()
        individuals = [ind[1] for ind in offspring]
    return individuals


def mutate_individuals(individuals,p=1):
    for ind in individuals:
        ind.mutate_genotypes(p=p)


def main(display=False, interval = 100, verbose_interval = 1):
    scores = []
    height = 500
    width = 500

    scale = 1
    scaled_h = int(height*scale)
    scaled_w = int(width*scale)
    duplication_interval = 1000
    removal_interval = 12000
    num_points = 1000
    num_individuals = 8
    itterations = 2500
    limit = 370
    ground_truth_unscaled = cv2.imread("girl.jpg")

    ground_truth = cv2.resize(ground_truth_unscaled, (scaled_w,scaled_h), interpolation = cv2.INTER_AREA)
    individuals = [Individual(num_points,scaled_h,scaled_w,ground_truth) for _ in range(num_individuals)]
    l_t = linkage_tree(individuals)

    evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h)
    for i in range(itterations):
        if i !=0 and i%duplication_interval == 0 and i <= limit:
            duplication_event(individuals)
            evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h)
            l_t = linkage_tree(individuals)

        if i !=0 and i%removal_interval == 0 and i <= limit:
            removal_event(individuals)
            evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h)
            l_t = linkage_tree(individuals)


        random.shuffle(individuals)
        individuals = variation(individuals,l_t,ground_truth,scaled_w,scaled_h)
        individuals = tournament_selection(individuals)
        individuals.sort(key=lambda x: x.score)
        l_t = linkage_tree(individuals)
        scores.append("%.2f" % individuals[0].score)


        if display and i % interval == 0:

            cv2.imshow('', individuals[0].painting)
            cv2.waitKey(1)
            name = "Img_"+str(i)+".jpg"
            cv2.imwrite(".\Ouput\_"+name,cv2.cvtColor(individuals[0].painting, cv2.COLOR_RGB2GRAY))
        if i % verbose_interval == 0:
            print("Best score for itteration", i,  individuals[0].score, " :length of genotypes", len(individuals[0].place_genotype)/2 )

    with open('data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, doublequote= False)
        wr.writerow(scores)


    cv2.imshow('', individuals[0].painting)
    cv2.waitKey(0)

if __name__ == "__main__":

    #cProfile.run('main(display=True, interval=1)')
    main(display=True, interval=1)


