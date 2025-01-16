import numpy
import numpy as np
import cv2

import cProfile
from imgcompare import image_diff
from PIL import Image, ImageDraw
import csv
import  copy
import random
import multiprocessing as mp
def make_veroni_points(height,width,num_points):
    veroni_points = np.asarray([np.random.randint(0,[height,width]) for _ in range(num_points) ])
    veroni_colors =  np.asarray([np.random.randint(256,size=3) for _ in range(num_points) ])
    return veroni_points,veroni_colors



class Individual(object):
    def __init__(self, num_points, height, width):
        self.place_genotype_c = np.random.choice(range(height), p=np.ones(height) /height, size=3 * num_points)
        self.color_genotype_c = np.random.choice(range(256), p=np.ones(256) / 256, size=4 * num_points)
        self.score = np.inf
        self.height = height
        self.width = width



    def make_p_genotype(self, num_points, height, width):
        x = np.random.choice(range(height + 1), p=np.ones(height + 1) / (height + 1), size=num_points)
        y = np.random.choice(range(width + 1), p=np.ones(width + 1) / (width + 1), size=num_points)
        place_g = np.append(x, y, axis=np.newaxis)
        return place_g

    def mutate_genotypes(self, p=0.02):
        max_xy = self.height -1
        max_c = 256
        mutations = ['shift', 'color']
        weights = [2/4,2/4]
        strenth = np.random.random()*2
        choice = np.random.choice(mutations,p=weights)

        if choice == 'shift':
            mask_p = np.random.choice([True, False], size=len(self.place_genotype_c), p=[p, 1 - p])
            mutation_p = np.random.uniform(low=-(strenth/2), high=(strenth/2),size=len(self.place_genotype_c)) *max_xy
            mutation_p = mutation_p + self.place_genotype_c
            mutation_p = np.where(mutation_p< 0, self.place_genotype_c, mutation_p)
            mutation_p = np.where(mutation_p > max_xy , self.place_genotype_c, mutation_p)
            self.place_genotype_c = np.where(mask_p,mutation_p.astype(int),self.place_genotype_c)



        if choice == 'color':
            mask_c = np.random.choice([True, False], size=len(self.color_genotype_c), p=[p, 1 - p])
            mutation_c = np.random.uniform(low=-(strenth / 2), high=(strenth / 2), size=len(self.color_genotype_c)) * max_c
            mutation_c = mutation_c + self.color_genotype_c
            mutation_c = np.where(mutation_c < 0, self.color_genotype_c, mutation_c)
            mutation_c = np.where(mutation_c > 255, self.color_genotype_c, mutation_c)
            self.color_genotype_c = np.where(mask_c,mutation_c.astype(np.uint8),self.color_genotype_c)






    def get_params(self):
        return self.place_genotype_c.reshape((-1, 3)), self.color_genotype_c.reshape((-1, 4))


    def duplicate_genome(self):
        num_points = len(self.place_genotype_c) // 3

        split_c_c = np.split(self.color_genotype_c, num_points)
        split_p_c = np.split(self.place_genotype_c, num_points)

        self.place_genotype_c = np.array([])
        self.color_genotype_c = np.array([])
        for i in range(num_points):

            self.place_genotype_c = np.append(self.place_genotype_c, np.append(split_p_c[i], split_p_c[i]))
            self.color_genotype_c = np.append(self.color_genotype_c, np.append(split_c_c[i], split_c_c[i]))





def evaluate_individuals(individuals, ground_t,width,height, trasnparent):
    for ind in individuals:
        places,colors = ind.get_params()
        if trasnparent:
            im = draw_circles_shaded(places,colors,width,height)
        else:
            painting = draw_circle(places,colors,width,height)
            im = Image.fromarray(painting)
        ind.score = image_diff(ground_t, im)

def evaluate_individual(ind, ground_t,width,height, trasnparent):
    places,colors = ind.get_params()
    if trasnparent:
        im = draw_circles_shaded(places,colors,width,height)
    else:
        painting = draw_circle(places,colors,width,height)
        im = Image.fromarray(painting)
    ind.score = image_diff(ground_t, im)


def gene_optimal_mixing(individuals,ground_t,width, height,x,i = 0,asynch = False):
    o_i = copy.deepcopy(x)
    best = copy.deepcopy(x)
    parents = copy.deepcopy(individuals)
    mutate_individuals(parents,p=0.75)

    for j in range(int(len(o_i.place_genotype_c)/3)):
        p = np.random.choice(parents)
        o_i.place_genotype_c[3*j:3*j+3] = p.place_genotype_c[3*j:3*j+3]
        o_i.color_genotype_c[4*j:4*j + 4] = p.color_genotype_c[4*j:4*j+ 4]

        evaluate_individual(o_i,ground_t=ground_t,width=width,height=height,trasnparent=True)
        if o_i.score < best.score:
            best.place_genotype_c[3 * j:3 * j + 3] = o_i.place_genotype_c[3 * j:3 * j + 3]
            best.color_genotype_c[4 * j:4 * j + 4] = o_i.color_genotype_c[4 * j:4 * j + 4]

            best.score = o_i.score

        else:
            o_i.place_genotype_c[3 * j:3 * j + 3] = best.place_genotype_c[3 * j:3 * j + 3]
            o_i.color_genotype_c[4 * j:4 * j + 4] = best.color_genotype_c[4 * j:4 * j + 4]

            o_i.score = best.score

    if not asynch:
        return o_i
    else: return (i,o_i)

def variation(individuals,ground_t,width,height,type='GOM'):
    offspring = []

    if type == 'GOM':
        for individual in individuals:
            offspring.append(gene_optimal_mixing(individuals, ground_t, width, height, individual))
        individuals = offspring

    if type == "parralel_GOM":
        pool = mp.Pool(mp.cpu_count())
        offspring = pool.starmap_async(gene_optimal_mixing, [( individuals,ground_t, width, height, individual, i , True) \
                                                   for i, individual in enumerate(individuals)]).get()
        pool.close()
        individuals = [ind[1] for ind in offspring]



    return individuals


def mutate_individuals(individuals,p=0.0075):
    for ind in individuals:
        ind.mutate_genotypes(p=p)

def tournament_selection(population, offspring):
    selection_pool = np.concatenate((population, offspring), axis=None)
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


def draw_circle(genotype_coords, genotype_colors, img_width, img_height):
    canvas = numpy.ones((img_height,img_width,3),dtype=np.uint8)*255

    for i,circles in enumerate(genotype_coords):
        pts = np.array([circles[0],circles[1]],dtype=int)

        RED =(int(genotype_colors[i,0]),int(genotype_colors[i,1]),int(genotype_colors[i,2]))
        cv2.circle(canvas,pts,int(circles[2]/10),color=RED,thickness= -1)
    return canvas

def draw_circles_shaded(genotype_coords, genotype_colors, img_width, img_height):
    img = Image.new('RGB', (img_width, img_height),color = (255,255,255))
    drw = ImageDraw.Draw(img, 'RGBA')

    for i,circle in enumerate(genotype_coords):
        pts =[circle[0]-(circle[2]/10), circle[1]-(circle[2]/10), circle[0]+(circle[2]/10), circle[1]+(circle[2]/10)]
        RED =(int(genotype_colors[i,2]),int(genotype_colors[i,1]),int(genotype_colors[i,0]),int(genotype_colors[i,3]))
        drw.ellipse(xy=pts, fill=RED)
    del drw
    return img

def duplication_event(individuals ):
    for ind in individuals:
        ind.duplicate_genome()

def main(display=False, interval = 100, verbose_interval = 1, transparent = False):
    scores = []
    height = 500
    width = 500
    scale = 0.5
    scaled_h = int(height*scale)
    scaled_w = int(width*scale)
    duplication_interval = 400
    num_points = 200
    num_individuals = 8
    itterations = 715
    limit = 4000
    ground_truth_unscaled = cv2.imread('van_goh.jpg')
    individuals = [Individual(num_points,scaled_h,scaled_w) for _ in range(num_individuals)]
    ground_truth_array = cv2.resize(ground_truth_unscaled, (scaled_w,scaled_h), interpolation = cv2.INTER_AREA)
    ground_truth = Image.fromarray(ground_truth_array)
    evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h, trasnparent=transparent)
    for i in range(itterations):
        if i !=0 and i%duplication_interval == 0 and i <= limit:
            duplication_event(individuals)
            evaluate_individuals(individuals, ground_truth, scaled_w, scaled_h,transparent)

        individuals = variation(individuals, ground_truth, scaled_w, scaled_h)
        individuals = tournament_selection(individuals,[])
        individuals.sort(key=lambda x: x.score)

        scores.append(individuals[0].score)
        if display and i % interval == 0:
            places,colors = individuals[0].get_params()
            if not transparent:
                painting = draw_circle(places*(1/scale),colors , width, height)
            if transparent:
                painting = np.asarray(draw_circles_shaded(places*(1/scale),colors , width, height))

            cv2.imshow('', painting)
            cv2.waitKey(1)
            name = "Img_" + str(i) + ".jpg"
            cv2.imwrite("./Ouput/_" + name, painting)
        if i % verbose_interval == 0:
            print("Best score for itteration", i,  individuals[0].score, " :length of genotypes", len(individuals[0].place_genotype_c)/6 )

    with open('data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE, doublequote= False)
        wr.writerow(scores)



if __name__ == "__main__":
    main(display=True, interval=1, transparent=True, verbose_interval=1)