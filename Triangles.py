import random

import numpy
import numpy as np
import cv2

import cProfile
from imgcompare import image_diff
from PIL import Image, ImageDraw
import csv
import  copy
import random
def make_veroni_points(height,width,num_points):
    veroni_points = np.asarray([np.random.randint(0,[height,width]) for _ in range(num_points) ])
    veroni_colors =  np.asarray([np.random.randint(256,size=3) for _ in range(num_points) ])
    return veroni_points,veroni_colors



class Individual(object):
    def __init__(self, num_points, height, width):
        self.place_genotype = np.random.choice(range(height), p=np.ones(height) /height, size=6 * num_points)
        self.color_genotype = np.random.choice(range(256), p=np.ones(256) / 256, size=4 * num_points)
        self.score = np.inf
        self.height = height
        self.width = width



    def make_p_genotype(self, num_points, height, width):
        x = np.random.choice(range(height + 1), p=np.ones(height + 1) / (height + 1), size=num_points)
        y = np.random.choice(range(width + 1), p=np.ones(width + 1) / (width + 1), size=num_points)
        place_g = np.append(x, y, axis=np.newaxis)
        return place_g

    def mutate_genotypes(self, p=0.01):
        max_xy = self.height -1
        max_c = 256
        mutations = ['shift', 'color']
        weights = [2/4,2/4]
        strenth = np.random.random()*4
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
        return self.place_genotype.reshape((-1, 6)), self.color_genotype.reshape((-1, 4))


    def duplicate_genome(self):
        self.place_genotype = np.append(self.place_genotype,copy.deepcopy(self.place_genotype))
        self.color_genotype = np.append(self.color_genotype, copy.deepcopy(self.color_genotype))

def evaluate_image(base, img):
    assert base.shape == img.shape, "Images are not the same shape"
    error = np.mean((base - img)**2)
    return error



def uniform_crossover(individual_a: Individual, individual_b: Individual, p=0.50    ):
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

    offspring_a.mutate_genotypes()
    offspring_b.mutate_genotypes()
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

def make_offspring(individuals):
    offspring = []
    for i in range(0, len(individuals), 2):
        ind_1 = individuals[i]
        ind_2 = individuals[i + 1]
        off_1, off_2 = uniform_crossover(ind_1, ind_2)
        offspring.append(off_1)
        offspring.append(off_2)
    return offspring

def evaluate_individuals(individuals, ground_t,width,height, trasnparent ):
    for ind in individuals:
        veroni_points, veroni_colors = ind.get_params()
        if trasnparent:
            im = draw_triangles_shaded(veroni_points,veroni_colors,width,height)
        else:
            painting = draw_triangles(veroni_points,veroni_colors,width,height)
            im = Image.fromarray(painting)
        ind.score = evaluate_image(ground_t,im)

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


def draw_triangles(genotype_coords, genotype_colors, img_width, img_height):
    canvas = numpy.ones((img_height,img_width,3),dtype=np.uint8)*255

    for i,triangle in enumerate(genotype_coords):
        pts = np.array([[triangle[0],triangle[1]],[triangle[2],triangle[3]],[triangle[4],triangle[5]]],dtype=int)

        RED =(int(genotype_colors[i,0]),int(genotype_colors[i,1]),int(genotype_colors[i,2]))
        cv2.fillPoly(img=canvas,pts=[pts],color=RED)
    return canvas

def draw_triangles_shaded(genotype_coords, genotype_colors, img_width, img_height):
    img = Image.new('RGB', (img_width, img_height),color = (255,255,255))
    drw = ImageDraw.Draw(img, 'RGBA')

    for i,triangle in enumerate(genotype_coords):
        pts = [(triangle[0],triangle[1]),(triangle[2],triangle[3]),(triangle[4],triangle[5])]
        RED =(int(genotype_colors[i,2]),int(genotype_colors[i,1]),int(genotype_colors[i,0]),int(genotype_colors[i,3]))
        drw.polygon(xy=pts, fill=RED)
    del drw
    return img

def duplication_event(individuals ):
    for ind in individuals:
        ind.duplicate_genome()

def main(display=False, interval = 1, verbose_interval = 1, transparent = True):
    scores = []
    height = 500
    width = 500
    scale = 1
    scaled_h = int(height*scale)
    scaled_w = int(width*scale)
    duplication_interval = 4000
    num_points = 200
    num_individuals = 24
    itterations = 8000
    limit = 4000
    ground_truth_unscaled = cv2.imread('pixl.jpg')
    individuals = [Individual(num_points,scaled_h,scaled_w) for _ in range(num_individuals)]
    ground_truth_array = cv2.resize(ground_truth_unscaled, (scaled_w,scaled_h), interpolation = cv2.INTER_AREA)
    #if transparent:
        #ground_truth_array = cv2.cvtColor(ground_truth_array,cv2.COLOR_BGR2RGB)

    ground_truth = Image.fromarray(ground_truth_array)
    evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h, trasnparent=transparent)
    for i in range(itterations):
        if i !=0 and i%duplication_interval == 0 and i <= limit:
            duplication_event(individuals)
            evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h,transparent)
        random.shuffle(individuals)
        offspring= make_offspring(individuals)
        evaluate_individuals(offspring, ground_truth, scaled_w, scaled_h,trasnparent=transparent)
        individuals = tournament_selection(individuals,offspring)
        individuals.sort(key=lambda individual: individual.score)
        scores.append(individuals[0].score)
        if display and i % interval == 0:
            veroni_points, veroni_colors = individuals[0].get_params()
            if not transparent:
                painting = draw_triangles(veroni_points*(1/scale), veroni_colors, width, height)
            if transparent:
                painting = np.asarray(draw_triangles_shaded(veroni_points * (1 / scale), veroni_colors, width, height))

            cv2.imshow('', painting)
            cv2.waitKey(1)
            name = "Img_" + str(i) + ".jpg"
            cv2.imwrite("./Ouput/_" + name, painting)
        if i % verbose_interval == 0:
            print("Best score for itteration", i,  individuals[0].score, " :length of genotypes", len(individuals[0].place_genotype)/6 )

    with open('data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, doublequote= False)
        wr.writerow(scores)


    veroni_points,veroni_colors = individuals[0].get_params()
    painting = draw_triangles(veroni_points*(1/scale),veroni_colors,width,height)
    cv2.imshow('', painting)
    cv2.waitKey(0)

if __name__ == "__main__":
    main(display=True,verbose_interval=1, transparent=False)