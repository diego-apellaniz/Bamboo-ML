#!/usr/bin/env python
# -*- coding: utf-8 -*-

# we will define all classes here regarding the objects we need to define our environment
# length units are cm

import pandas as pd
import copy
from bisect import bisect_left, bisect_right
import re
import numpy as np

#import sys
#sys.setrecursionlimit(1500)

# function to check if item is contained in list of intervals
def contained(a, interval_list):
    return any(a in x for x in interval_list)

class Culm:
    def __init__(self, id, d1, d2, t1, t2, nodes):
        self.id = id
        self.d1 = d1 # diameter 1
        self.d2 = d2 # diameter 2
        self.t1 = t1 # length before first node
        self.t2 = t2 # length after last node
        self.nodes = [n + t1 for n in nodes]
        self.length = nodes[-1] + t1+ t2
        self.available = [pd.Interval(0, self.length, closed='both')]
        self.used = False # has this culm already been used for any member?
    def check(self, member):
        parameter_of_solutions = []
        scores = []
        # create matrix of member requirements -> where the connections can be
        # rows -> nodes in culm
        # columns -> target item to check
        target_space = [[pd.Interval(node-member.s0[i], node+member.s1[i], closed='both') for node in self.nodes] for i in range(len(member.s))]
        target_positions = []        
        for interval in target_space[1]:
            target_positions.extend(range(interval.left, interval.right))
        for interval in target_space[0]:
            target_positions.extend(range(interval.left, interval.right +1))
        target_positions.sort()
        #print(target_space)
        #print()
        #member_check = []
        #for s in member.s:
        #    member_check.append(any(contained(s, row) for row in target_space))
        #check = all(member_check)
        # check if all member requirements are satisfied and if the whole member is located in of the remaining culm pieces
        #for t in range(self.length - member.s[-1] + 1):
        #for i in [i for i in range(len(members)) if not members_found[0,i]]:
        positions_to_check = [t for t in target_positions if t <= self.length - member.s[-1]]
        for t in positions_to_check:
            if all(contained(member.s[i]+t, target_space[i]) for i in range(len(member.s))):
                for culm_piece in self.available:
                    if member.s[0]+t in culm_piece and member.s[-1]+t in culm_piece:
                        # EDIT SCORE!!!!!
                        score = 2*(max(t, self.length - member.length -t)/(self.length-member.length)-0.5) # in order to maximize maximum length of remainin available culm -> r = L_greatest_part / (1 - L_solution) - from 0.0 to 1.0
                        # print(f"Solution found at t = {t}. Score = {score}")
                        parameter_of_solutions.append(t)
                        scores.append(score)
        return parameter_of_solutions, scores
    def availabletotext(self):
        textinterval = ""
        for inter in self.available:
            textinterval += f"from {inter.left} to {inter.right}; "            
        return textinterval
    def extractSolution(self, t0, member):
        if t0+member.length>self.length:
            return False
        start_node_index = bisect_left(self.nodes, t0 + 0.05)-1
        end_node_index = bisect_right(self.nodes, t0 + member.length -0.05) # -0.05 to avoid out of bounds
        solution_culm = copy.deepcopy(self)
        solution_culm.available = [pd.Interval(self.nodes[start_node_index]-5, self.nodes[end_node_index]+5, closed='both')]
        solution = Solution(solution_culm, member.id, t0)        
        new_available_interval=[]
        for interval in self.available:
            if not t0 in interval:
                new_available_interval.append(interval)
            else:
                start_node_interval_index = bisect_right(self.nodes, interval.left)
                end_node_interval_index = bisect_left(self.nodes, interval.right)-1
                if start_node_index - start_node_interval_index > 1: # otherwise that remaining part is uselss
                    new_available_interval.append(pd.Interval(self.nodes[start_node_interval_index]-5, self.nodes[start_node_index-1]+5, closed='both'))
                if end_node_interval_index - end_node_index > 1: # otherwise that remaining part is uselss
                    new_available_interval.append(pd.Interval(self.nodes[end_node_index+1]-5, self.nodes[end_node_interval_index]+5, closed='both'))
                reward = solution_culm.available[0].length / interval.length # reward shows the utilization you made of the available culm
                #reward = solution_culm.available[0].length / self.length # aöternative and more simple reward -> needed in order not to consider the order of members chosen
                #print(f"(reward = {reward}")
        self.available = new_available_interval
        return solution, reward
    def export(self, episode, move):
        interval_list = []
        for interval in self.available:
            interval_list.append(interval.left)
            interval_list.append(interval.right)
        inervals_text = str(interval_list)
        #inervals_text = inervals_text[1:len(inervals_text)-3]
        inervals_text = re.sub('[\[\] ]', '', inervals_text)
        nodes_text = str(self.nodes)
        #nodes_text = nodes_text[1:len(nodes_text)-3]
        nodes_text = re.sub('[\[\] ]', '', nodes_text)
        culm_text = f"culm;{episode};{move};{self.id};{self.d1};{nodes_text};{inervals_text}"
        return culm_text       

class Member:
    def __init__(self, id, s, s0, s1):
        self.id = id 
        self.s0 = s0 # tolerance before node
        self.s1 = s1 # tolerance after node
        self.s = s # coordinates of the desired nodes
        if len(s) != len(s0) or len(s) != len(s1):
            raise ValueError('Member is not valid')
        self.length = s[-1]-s[0]
    def get_culm(self, solutions):
        for solution in solutions:
            if self.id == solution.member_id:
                return solution.culm.id +1 # to differentiate it from 0!!
        return 0
    
class Solution:
    def __init__(self, culm, member_id, t0):
        self.culm = culm
        self.member_id = member_id
        self.t0 =  t0
    def export(self, episode, move):
        interval_list = []
        for interval in self.culm.available:
            interval_list.append(interval.left)
            interval_list.append(interval.right)
        inervals_text = str(interval_list)
        #inervals_text = inervals_text[1:len(inervals_text)-3]
        inervals_text = re.sub('[\[\] ]', '', inervals_text)
        nodes_text = str(self.culm.nodes)
        #nodes_text = nodes_text[1:len(nodes_text)-3]
        nodes_text = re.sub('[\[\] ]', '', nodes_text)
        culm_text = f"solution;{episode};{move};{self.culm.id};{self.member_id};{self.culm.d1};{self.t0};{nodes_text};{inervals_text}"
        return culm_text

class StructureEnv:
    #def __init__(self):
    #    self.current_episode = 0
    def reset(self, culms, members, max_moves):
        self.max_moves = max_moves
        self.culms = copy.deepcopy(culms)
        self.members = copy.deepcopy(members)
        self.solutions = []
        self.move = 0
        #self.members_found = np.zeros((1, len(members)), dtype="int8")
        self.members_found = np.zeros((len(culms), len(members)), dtype="int8")        
        self.current_culm=0
        self.current_member=0
        self.t_split = 0.0
    def find_next_state(self):
        #if self.members_found.all():
        if self.structure_complete():
            new_state = self.extend_current_state([0, 0]) # we need some value so that the network doesn´t return an error
            done = True
            untouched_bamboos = [x for x in self.culms if len(x.available) and x.length == x.available[0].length]
            #extra_reward = len(untouched_bamboos)
            extra_reward=0            
        else:
            t, scores = self.culms[self.current_culm].check(self.members[self.current_member]) 
            #if len(t)>0 and self.move < self.max_moves and self.members_found[0][self.current_member] == 0:
            if len(t)>0 and self.move < self.max_moves and self.members_found[self.current_culm][self.current_member] == 0:
                done = False
                self.t_split = t[np.argmax(scores)]
                new_state = self.extend_current_state([self.current_member, self.current_culm])
                done = False
                extra_reward=0
            elif self.move < self.max_moves:
                #recursion to finde next_state->
                self.switch_to_next_culm()                
                new_state, done, extra_reward = self.find_next_state()
            else:
                new_state = self.extend_current_state([0, 0]) # we need some value so that the network doesn´t return an error
                done = True
                unsatisfied_members = [x for x in range(len(self.members)) if not 1 in self.members_found[:,x]]
                extra_reward = - len(unsatisfied_members)
        return new_state, done, extra_reward
    def extract_solution(self):
        solution, reward = self.culms[self.current_culm].extractSolution(self.t_split, self.members[self.current_member])
        self.solutions.append(solution)
        self.members_found[self.current_culm][self.current_member] = 1
        return reward
    def switch_to_next_culm(self):        
        if self.members_found[self.current_culm][self.current_member] == 1:            
            while self.members_found[self.current_culm][self.current_member % len(self.members)] == 1:
                self.moves_same_member = 0
                self.current_member +=1
                self.move += 1
                if self.move >= self.max_moves:
                    return # exit loop if maximum number of moves exceeded -> no possible solution
            self.current_member = self.current_member % len(self.members) # In case that the number of members is exceeded          
            self.current_culm =0
        else:
            if self.current_culm == len(self.culms):
                self.current_member +=1
            self.current_culm = (self.current_culm + 1) % len(self.culms)
            self.move += 1
    def export(self, episode, step, reward, action):
        myjson = jsonfile()
        myjson.episode = episode
        myjson.step = step
        myjson.reward = reward
        myjson.action = action
        myjson.current_culm = self.current_culm
        myjson.current_member = self.current_member
        culms = {}
        for culm in self.culms:
            culms[culm.id] = culm.export(episode,self.move)
        myjson.culms = culms
        solutions = {}
        for solution in self.solutions:
            solutions[solution.member_id] = solution.export(episode, self.move)
        myjson.solutions = solutions
        return myjson.__dict__

    def structure_complete(self):
        for member in range(len(self.members)):
            if not 1 in self.members_found[:,member]:
                return False
        return True

    def extend_current_state(self, current_state):
        # Calculate new state with consideration of the array of found members
        member_culms = [member.get_culm(self.solutions) for member in self.members]
        current_state.extend(member_culms)
        return current_state

class jsonfile ():
    def __init__(self):
        self.episode = 0
        self.step = 0
        self.reward = 0
        self.action = 0
        self.current_member = 0
        self.current_culm = 0
        self.culms = {}
        self.solutions = {}
        