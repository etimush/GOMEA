import numpy as np
import cv2
from numba import jit
#from joblib import Parallel
from scipy.spatial import KDTree
from scipy.stats import chi2_contingency
import time
import cProfile
from imgcompare import image_diff
from PIL import Image
import csv
import  copy
import random
def make_veroni_points(height,width,num_points):
    veroni_points = np.asarray([np.random.randint(0,[height,width]) for _ in range(num_points) ])
    veroni_colors =  np.asarray([np.random.randint(256,size=3) for _ in range(num_points) ])
    return veroni_points,veroni_colors


class Individual(object):
    def __init__(self, num_points, height, width):


        self.place_genotype = self.make_p_genotype(num_points, height, width)
        self.color_genotype = np.random.choice(range(256), p=np.ones(256) / 256, size=3 * num_points)
        self.score = np.inf
        self.height = height
        self.width = width



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

        if choice == 'shift':
            mask_p = np.random.choice([True, False], size=len(self.place_genotype), p=[p, 1 - p])
            mutation_p = np.random.uniform(low=-(strenth/2), high=(strenth/2),size=len(self.place_genotype)) *max_xy
            mutation_p = mutation_p + self.place_genotype
            mutation_p = np.where(mutation_p< 0, self.place_genotype, mutation_p)
            mutation_p = np.where(mutation_p > max_xy , self.place_genotype, mutation_p)
            self.place_genotype = np.where(mask_p,mutation_p.astype(int),self.place_genotype)

        if choice == 'color':
            mask_c = np.random.choice([True, False], size=len(self.color_genotype), p=[p, 1 - p])
            mutation_c = np.random.uniform(low=-(strenth / 2), high=(strenth / 2), size=len(self.color_genotype)) * max_c
            mutation_c = mutation_c + self.color_genotype
            mutation_c = np.where(mutation_c < 0, self.color_genotype, mutation_c)
            mutation_c = np.where(mutation_c > 255, self.color_genotype, mutation_c)
            self.color_genotype = np.where(mask_c,mutation_c.astype(np.uint8),self.color_genotype)




    def get_params(self):
        return self.place_genotype.reshape((-1, 2)), self.color_genotype.reshape((-1, 3))


    def duplicate_genome(self):

        self.color_genotype = np.append(self.color_genotype,copy.deepcopy(self.color_genotype))
        shift_1 = [ np.random.uniform(low= -0.01, high= 0.01) for _ in range(len(self.place_genotype))]
        shift_2 = [ np.random.uniform(low= -0.01, high= 0.01) for _ in range(len(self.place_genotype))]
        self.place_genotype = np.append(self.place_genotype +shift_1 , copy.deepcopy(self.place_genotype)-shift_1)


    def removal_event(self):
        num = range(int((len(self.place_genotype)/2) -1))
        places = np.random.choice(num,size=int(0.05*len(self.place_genotype)))
        p1 = lambda x: 2*x +1
        p2 = lambda x: 2*x +2

        c1 = lambda x: 3*x +1
        c2 = lambda x: 3 * x + 2
        c3 = lambda x: 3 * x + 3
        p_i = [ps(place) for place in places for ps in (p1,p2)]
        c_i = [ps(place) for place in places for ps in (c1,c2,c3)]

        self.place_genotype = np.delete(self.place_genotype,p_i)
        self.color_genotype = np.delete(self.color_genotype, c_i)



def evaluate_image(base, img):
    assert base.shape == img.shape, "Images are not the same shape"
    error = np.sum(np.abs(base - img))
    return error



def uniform_crossover(individual_a: Individual, individual_b: Individual, error, p=0.50,):
    offspring_a = copy.deepcopy(individual_a)
    offspring_b = copy.deepcopy(individual_b)

    min_c = min(len(individual_a.color_genotype),len(individual_b.color_genotype))
    min_p = min(len(individual_a.place_genotype),len(individual_b.place_genotype))

    list_rand_p = np.random.rand(min_p)
    list_rand_c = np.random.rand(min_c)

    offspring_a.place_genotype[:min_p] = np.where(list_rand_p >= p, individual_b.place_genotype[:min_p], individual_a.place_genotype[:min_p])
    offspring_b.place_genotype[:min_p] = np.where(list_rand_p >= p, individual_a.place_genotype[:min_p], individual_b.place_genotype[:min_p])


    offspring_a.color_genotype[:min_c] = np.where(list_rand_c >= p, individual_b.color_genotype[:min_c], individual_a.color_genotype[:min_c])
    offspring_b.color_genotype[:min_c] = np.where(list_rand_c >= p, individual_a.color_genotype[:min_c], individual_b.color_genotype[:min_c])

    offspring_a.mutate_genotypes(p = 0.0075*error)
    offspring_b.mutate_genotypes(p = 0.0075*error)
    return offspring_a, offspring_b

def one_point_crossover(individual_a: Individual, individual_b: Individual):
    offspring_a = copy.deepcopy(individual_a)
    offspring_b = copy.deepcopy(individual_b)
    random_place_p = np.random.randint(0,len(individual_a.place_genotype))
    random_place_c = np.random.randint(0,len(individual_a.color_genotype))

    offspring_a.place_genotype = np.append(individual_a.place_genotype[:random_place_p],individual_b.place_genotype[random_place_p:])
    offspring_b.place_genotype = np.append(individual_b.place_genotype[:random_place_p],
                                           individual_a.place_genotype[random_place_p:])

    offspring_a.color_genotype = np.append(individual_a.color_genotype[:random_place_c],
                                           individual_b.color_genotype[random_place_c:])
    offspring_b.color_genotype = np.append(individual_b.color_genotype[:random_place_c],
                                           individual_a.color_genotype[random_place_c:])
    offspring_a.mutate_genotypes()
    offspring_b.mutate_genotypes()
    return offspring_a, offspring_b

def three_point_uniform_crossover(individual_a: Individual, individual_b: Individual, p=0.25, linked = False):
    offspring_a = copy.deepcopy(individual_a)
    offspring_b = copy.deepcopy(individual_b)

    num = range(int((len(individual_a.place_genotype) / 2) - 1))
    places = np.random.choice(num, size=int(p * len(individual_a.place_genotype)))

    p1 = lambda x: 2 * x + 1
    p2 = lambda x: 2 * x + 2

    c1 = lambda x: 3 * x + 1
    c2 = lambda x: 3 * x + 2
    c3 = lambda x: 3 * x + 3
    p_i = [ps(place) for place in places for ps in (p1, p2)]
    c_i = [ps(place) for place in places for ps in (c1, c2, c3)]

    p_t = [True if i in p_i else False for i in range(len(individual_a.place_genotype))]
    c_t = [True if i in c_i else False for i in range(len(individual_a.color_genotype))]

    if linked:
        offspring_a.place_genotype = np.where(p_t, individual_b.place_genotype, individual_a.place_genotype)
        offspring_b.place_genotype = np.where(p_t , individual_a.place_genotype, individual_b.place_genotype)

        offspring_a.color_genotype = np.where(c_t, individual_b.color_genotype, individual_a.color_genotype)
        offspring_b.color_genotype = np.where(c_t, individual_a.color_genotype, individual_b.color_genotype)
    else:
        swaps = ['place', 'color']
        weights = [2 / 4, 2 / 4]

        choice = np.random.choice(swaps, p=weights)
        if choice == 'place':
            offspring_a.place_genotype = np.where(p_t, individual_b.place_genotype, individual_a.place_genotype)
            offspring_b.place_genotype = np.where(p_t, individual_a.place_genotype, individual_b.place_genotype)

        if choice == 'color':
            offspring_a.color_genotype = np.where(c_t, individual_b.color_genotype, individual_a.color_genotype)
            offspring_b.color_genotype = np.where(c_t, individual_a.color_genotype, individual_b.color_genotype)

    offspring_a.mutate_genotypes(p = 0.0075)
    offspring_b.mutate_genotypes(p = 0.0075)

    return offspring_a, offspring_b

def make_offspring(individuals,error):
    offspring = []
    for i in range(0, len(individuals), 2):
        ind_1 = individuals[i]
        ind_2 = individuals[i + 1]
        off_1, off_2 = three_point_uniform_crossover(ind_1, ind_2)
        offspring.append(off_1)
        offspring.append(off_2)
    return offspring

def evaluate_individuals(individuals, ground_t,width,height):

    for ind in individuals:

        veroni_points, veroni_colors = ind.get_params()
        painting = draw_voronoi_matrix(veroni_points,veroni_colors,width,height)
        im = Image.fromarray(painting)
        ground_truth = Image.fromarray(ground_t)
        ind.score = image_diff(ground_truth, im)

def tournament_selection(population, offspring):
    selection_pool = np.concatenate((population, offspring), axis=None)
    tournament_size = 4
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

def introduce_variatey(individuals,num_points,scaled_h,scaled_w):
    individuals.sort(key=lambda individual: individual.score)
    new_pool = individuals[:int((3 / 4) * len(individuals))]
    for i in range(int((1 / 4) * len(individuals))):
        new_pool.append(Individual(num_points, scaled_h, scaled_w))
    return new_pool

def main(display=False, interval = 100, verbose_interval = 1, cull_interval=10010):
    scores = []
    height = 500
    width = 500
    scale = 0.1
    scaled_h = int(height*scale)
    scaled_w = int(width*scale)
    duplication_interval = 1000
    removal_interval = 20000
    num_points = 50
    num_individuals = 24
    itterations = 4000
    limit = 4800
    error_frac = 1
    ground_truth_unscaled = cv2.imread("pixl.jpg")
    individuals = [Individual(num_points,scaled_h,scaled_w) for _ in range(num_individuals)]
    ground_truth = cv2.resize(ground_truth_unscaled, (scaled_w,scaled_h), interpolation = cv2.INTER_AREA)

    for i in range(itterations):
        if i !=0 and i%duplication_interval == 0 and i <= limit:
            duplication_event(individuals)
            evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h)

        if i !=0 and i%removal_interval == 0 and i <= limit:
            removal_event(individuals)
            evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h)

        if i!=0 and i%cull_interval == 0 :
            individuals = introduce_variatey(individuals,num_points,scaled_h,scaled_w)

        random.shuffle(individuals)
        offspring= make_offspring(individuals,error_frac)
        evaluate_individuals(offspring, ground_truth, scaled_w, scaled_h)
        individuals = tournament_selection(individuals,offspring)
        individuals.sort(key=lambda individual: individual.score)
        scores.append(individuals[0].score)

        if display and i % interval == 0:
            veroni_points, veroni_colors = individuals[0].get_params()
            painting = draw_voronoi_matrix(veroni_points*(1/scale), veroni_colors, width, height)
            cv2.imshow('', painting)
            cv2.waitKey(1)
            name = "Img_"+str(i)+".jpg"
            cv2.imwrite(".\Ouput\_"+name,painting)
        if i % verbose_interval == 0:
            print("Best score for itteration", i,  individuals[0].score, " :length of genotypes", len(individuals[0].place_genotype)/2 )

    with open('data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, doublequote= False)
        wr.writerow(scores)


    veroni_points,veroni_colors = individuals[0].get_params()
    painting = draw_voronoi_matrix(veroni_points*(1/scale),veroni_colors,width,height)
    cv2.imshow('', painting)
    cv2.waitKey(0)

if __name__ == "__main__":
    cProfile.run('main(display=True, interval=50)')



#Linkage_Tree for GOMEA variant
    # for the entire population: take every point and pair it up with its euklidean distance closest neighour
        #link sets of links by distance
    #When doing crossover, we corsover every variable in the linkage tree
    #We keep the change only if its beneficial

#Pseudo code:
#For i in rang len fos sunset -1
    #pick random parenit
    #swap fos subset I for parent and individual
    #evaluat efitness
    #if fitness is better than best besft for
    #best fos changed to offspring
    #else offspring fos changes to best


#In gomea we discard the parents
