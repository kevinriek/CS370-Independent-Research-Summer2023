import numpy as np
import math 
import queue
import os
import random

class map_manager:
    def __init__(self, dimensions):
        self.Map = self.map_init(dimensions)
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.Units = []
        
        self.Teams = [Team(0), Team(1)]
        self.curr_team = 0
        self.turn_count = 0
   
        self.visited_tiles = []
        
    def __str__(self):
        string = ""
        for i in range(self.Map.shape[0]):
            for j in range(self.Map.shape[1]):
                string += str(self.Map[i,j])
            string += "\n"
        return string
                
    def map_init(self, dimensions):
        Map = np.empty([dimensions[0], dimensions[1]], dtype=Tile)
        for i in range(Map.shape[0]):
            for j in range(Map.shape[1]):
                Map[i,j] = Tile((i, j), None)
        return Map

    def reset_map(self):
        for i in range(self.Map.shape[0]):
            for j in range(self.Map.shape[1]):
                self.Map[i, j].unit_ref = None
        for team in self.Teams:
            team.units = []
        self.curr_team = 0
        self.turn_count = 0
                

    def find_movement(self, unit):
        assert(unit is not None)
        unit.curr_move = unit.max_move
        self.dijstrkas(unit)
    
    def dijstrkas(self, unit):
        prio_queue = queue.PriorityQueue()
        Map = self.Map
        
        Map[unit.pos].move_cost = 0
        prio_queue.put(Map[unit.pos])
        self.visited_tiles.append(Map[unit.pos])
        
        while(not prio_queue.empty()):
            curr_tile = prio_queue.get()
            curr_pos = curr_tile.pos
            curr_tile.visited = True
            for i in range(4):
                new_pos = tuple(np.add(curr_pos, self.directions[i]))
                if (new_pos[0] < 0 or new_pos[0] >= self.Map.shape[0] or
                    new_pos[1] < 0 or new_pos[1] >= self.Map.shape[1] or
                    (Map[new_pos].unit_ref is not None and Map[new_pos].unit_ref.Team == unit.Team) or
                    new_pos == unit.pos or Map[new_pos].visited):
                    pass
                else:
                    new_cost = Map[new_pos].tile_cost + curr_tile.move_cost
                    if (Map[new_pos].move_cost > new_cost and new_cost <= unit.max_move):
                        Map[new_pos].move_cost = new_cost
                        Map[new_pos].move_parent = curr_tile
                        if (Map[new_pos].unit_ref is not None):
                            Map[new_pos].is_attack = True
                        self.visited_tiles.append(Map[new_pos])
                        prio_queue.put(Map[new_pos])
                    
        
    def place_unit(self, pos, team):
        new_unit = Unit(pos=pos, hp=100, mmove=10, Att=20, Def=10, Team=team)
        self.Map[pos].unit_ref = new_unit
        self.Units.append(new_unit)
        self.Teams[team].units.append(new_unit)
        return new_unit
    

    def print_units(self):
        for unit in self.Units:
            print(unit)


    def clear_movement(self):
        for i in range(self.Map.shape[0]):
            for j in range(self.Map.shape[1]):
                self.Map[i,j].move_cost = float('inf')
                self.Map[i,j].visited = False
                self.Map[i,j].is_attack = False
        self.visited_tiles = []
        
    
    def move_unit(self, unit, pos):
        #move_pos = tuple(np.add(unit.pos, dir))
        move_pos = pos
        tile = self.Map[move_pos]
        try:
            assert(tile.visited)
        except AssertionError:
            print("Tile {0} not reachable by unit".format(move_pos))

        if(tile.is_attack == True):
            move_tile = tile.move_parent 
            self.Map[unit.pos].unit_ref = None
            unit.pos = move_tile.pos
            unit.curr_move -= move_tile.move_cost
            move_tile.unit_ref = unit
            
            self.combat(unit, tile.unit_ref)
        else:
            move_tile = tile
            self.Map[unit.pos].unit_ref = None
            unit.pos = move_tile.pos
            unit.curr_move -= move_tile.move_cost
            move_tile.unit_ref = unit


        self.clear_movement()
        
    def combat(self, att_unit, def_unit):
        #att_dmg = (def_unit.Def / att_unit.Att) * 20
        def_dmg = (att_unit.Att / def_unit.Def) * 100
        #att_unit.hp -= att_dmg
        def_unit.hp -= def_dmg
        
        if (def_unit.hp <= 0):
            self.Units.remove(def_unit)
            self.Teams[def_unit.Team].units.remove(def_unit)
            self.Map[def_unit.pos].unit_ref = None
            del def_unit
    
    """
    def Turn(self):
        if (self.curr_team == 0):
            self.curr_team = 1
        else:
            self.curr_team = 0
            self.turn_count += 1
            
        for unit in self.Units:
            if (unit.Team == self.curr_team):
                unit.curr_move = unit.max_move
    """
    
    #returns -1 if the game is not over, else returns the number of the winning team
    def game_result(self):
        if len(self.Teams[0].units) == 0:
            return 1
        if len(self.Teams[1].units) == 0:
            return 0
        return -1
    
class Tile:
    def __init__(self, pos, unit):
        self.pos = pos
        self.unit_ref = unit
        self.tile_cost = 1.0 #Cost to move onto this tile
        
        self.move_cost = float('inf')
        self.move_parent = None
        self.visited = False
        self.is_attack = False

    def __str__(self):
        if (self.visited):
            string = '\033[94m' #Blue
            endstring = '\033[0m'
        else:
            string = ''
            endstring = ''
        if (self.unit_ref == None):
            return string + "[ ]" + endstring
        else:
            return string + "[{}]".format(self.unit_ref.Team) + endstring
        
    def __lt__(self, other):
        return self.move_cost < other.move_cost
    
    def print_tile(self):
        return "Tile ({0}, {1}):\nUnit: {2}".format(self.pos[0], self.pos[1], self.unit_ref)
        
class Team:
    def __init__(self, num):
        self.team_num = num
        self.units = []

class Unit:
    def __init__(self, pos, hp, mmove, Att, Def, Team):
        self.pos = pos
        self.hp = hp
        
        self.max_move = mmove
        self.curr_move = mmove

        self.Att = Att
        self.Def = Def
        self.Team = Team
        
    def __str__(self):
        string = "Unit:\n\tpos: {0}\n\tcurr_move: {1}\n\thp: {2}\n".format(self.pos, self.curr_move, self.hp) 
        string += "\tmax_move: {0}\n\tAtt: {1}\n\tDef: {2}\n\tTeam: {3}\n".format(self.max_move, self.Att, self.Def, self.Team)
        return string