import numpy as np
import math 
import queue
import os
import random
import copy
from colorama import Fore, Back, Style 

class map_manager:
    forest_list = [(0, 2), (0, 3), (0, 4), (0, 5),
                   (1, 2), (1, 3), (1, 4), (1, 5), 
                   (2, 2), (2, 3), (2, 4), (2, 5),
                   (3, 2), (3, 3), (3, 4), (3, 5),
                   (11, 12), (11, 11), (11, 10), (11, 9),
                   (12, 12), (12, 11), (12, 10), (12, 9),
                   (13, 12), (13, 11), (13, 10), (13, 9),
                   (14, 12), (14, 11), (14, 10), (14, 9),
                   ]
    hills_list = [(7, 2), (8, 2), (9, 2), (10, 2),
                  (4, 12), (5, 12), (6, 12), (7, 12),
                  (5, 7), (6, 7), (7, 7), (8, 7),
                 ]
    
    def __init__(self, dimensions):
        self.Map = self.map_init(dimensions)
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.Units = []
        
        self.map_layouts = []
        self.layout_unit_count = 0

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
                terrain_name = 'Plains'
                if (i, j) in map_manager.forest_list:
                    terrain_name = 'Forest'
                if (i, j) in map_manager.hills_list:
                    terrain_name = 'Hills'
                Map[i,j] = Tile((i, j), None, terrain_name)
        return Map

    def reset_map(self):
        self.Units = []
        self.clear_movement()
        for i in range(self.Map.shape[0]):
            for j in range(self.Map.shape[1]):
                tile = self.Map[i, j]
                tile.unit_ref = None
        for team in self.Teams:
            team.units = []
            team.live_units = []
        self.curr_team = 0
        self.turn_count = 0
                

    def find_movement(self, unit):
        assert(unit is not None)

        self.clear_movement()
        #unit.curr_move = unit.max_move
        self.dijstrkas(unit)
        return self.visited_tiles
    
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
                    Map[new_pos].visited): #or new_pos == unit.pos 
                    pass
                else:
                    new_cost = Map[new_pos].tile_cost + curr_tile.move_cost
                    att_range = unit.range
                    #Prevent negative range. NOTE: melee range is 0, not 1
                    #Range bonus only applies if unit is ranged
                    if unit.range > 0:
                        if Map[unit.pos].terrain.range_mod + att_range > 0:
                            att_range += Map[unit.pos].terrain.range_mod
                        else:
                            att_range = 0

                    if (Map[new_pos].move_cost > new_cost and new_cost <= (unit.max_move + att_range)):
                        Map[new_pos].move_cost = new_cost
                        Map[new_pos].move_parent = curr_tile
                        Map[new_pos].rotation = Unit.move_to_rotation[self.directions[i]]

                        if (Map[new_pos].unit_ref is not None):
                            if unit.range > 0:
                                parent = curr_tile
                                for i in range(att_range + 1):
                                    if parent.move_parent is not None:
                                        parent = parent.move_parent
                                    else:
                                        break
                                Map[new_pos].move_parent = parent
                            Map[new_pos].is_attack = True
                            Map[new_pos].visited = True
                            self.visited_tiles.append(Map[new_pos])
                        else:
                            #Should prevent movement through opponents
                            prio_queue.put(Map[new_pos])
                            if (new_cost <= unit.max_move):
                                self.visited_tiles.append(Map[new_pos])
                        
                    
        
    def place_unit(self, pos, unit_type, team):
        new_unit = Unit(pos=pos, unit_type=unit_type, Team=team)
        self.Map[pos].unit_ref = new_unit
        self.Units.append(new_unit)
        self.Teams[team].units.append(new_unit)
        self.Teams[team].live_units.append(new_unit)
        return new_unit
    

    def print_units(self):
        for unit in self.Units:
            print(unit)


    def clear_movement(self):
        for i in range(self.Map.shape[0]):
            for j in range(self.Map.shape[1]):
                tile = self.Map[i,j]
                tile.move_cost = float('inf')
                tile.visited = False
                tile.is_attack = False
                tile.move_parent = None
                tile.rotation = -1
        self.visited_tiles = []
        
    
    def move_unit(self, unit, pos):
        if pos[0] < 0 or pos[1] < 0:
            assert(False)
        #move_pos = tuple(np.add(unit.pos, dir))
        if pos == unit.pos:
            return

        move_pos = pos
        tile = self.Map[move_pos]
        """        try:                 #CURRENTLY AM NOT SETTING THE VISITED STAT SO THIS DOES NOT WORK
            assert(tile.visited)
        except AssertionError:
            print("Tile {0} not reachable by unit".format(move_pos))"""

        #Change unit rotation before combat
        unit.rotation = self.Map[move_pos].rotation

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
        
        
        
        
    def sim_combat(self, att_unit, def_unit):
        att = att_unit.Att
        if Unit.is_flank(def_unit, att_unit):
            att *= 3

        defs = def_unit.Def
        defs *= self.Map[def_unit.pos].terrain.def_mod

        att_dmg = (defs / att) * 20
        def_dmg = (att / defs) * 25
        
        att_unit.temp_hp = att_unit.hp - att_dmg
        def_unit.temp_hp = def_unit.hp - def_dmg

        if (def_unit.temp_hp <= 0):
            def_unit.temp_hp = 0
        if (att_unit.temp_hp <= 0):
           att_unit.temp_hp = 0

    def reset_temp_hp(self):
        for unit in self.Units:
            unit.temp_hp = unit.hp

    def combat(self, att_unit, def_unit):
        att = att_unit.Att
        if Unit.is_flank(def_unit, att_unit):
            att *= 3

        defs = def_unit.Def
        defs *= self.Map[def_unit.pos].terrain.def_mod

        # print("Attack: {}".format(att))
        # print("Defense: {}".format(defs))

        att_dmg = (defs / att) * 20
        def_dmg = (att / defs) * 25

        att_unit.hp -= att_dmg
        def_unit.hp -= def_dmg

        def_unit.temp_hp = def_unit.hp
        att_unit.temp_hp = att_unit.hp

        if (def_unit.hp <= 0):
            def_unit.hp = 0
            def_unit.temp_hp = def_unit.hp
            
            self.Teams[def_unit.Team].live_units.remove(def_unit)
            self.Map[def_unit.pos].unit_ref = None
            #del def_unit

        if (att_unit.hp <= 0):
            att_unit.hp = 0
            att_unit.temp_hp = att_unit.hp
            
            self.Teams[att_unit.Team].live_units.remove(att_unit)
            self.Map[att_unit.pos].unit_ref = None
    

    def Turn(self):
        if (self.curr_team == 0):
            self.curr_team = 1
        else:
            self.curr_team = 0
            self.turn_count += 1
            
        #for unit in self.Units:
        #    if (unit.Team == self.curr_team):
        #        unit.curr_move = unit.max_move

    def setup_layouts_rand(self, layout_n, unit_count):
        self.map_layouts = []
        dimensions = self.Map.shape
        
        for n in range(layout_n):
            pos_ls = []
            for i in range(unit_count):
                pos_ls.append((round(random.uniform((dimensions[0]/unit_count)*i, (dimensions[1]/unit_count)*(i+1)-1)),
                                round(random.uniform(0, dimensions[1]/7))))
            for i in range(unit_count):
                pos_ls.append((round(random.uniform((dimensions[0]/unit_count)*i, (dimensions[1]/unit_count)*(i+1)-1)),
                                round(random.uniform(dimensions[1]-(1+dimensions[1]/7), dimensions[1]-1))))

            self.map_layouts.append(pos_ls)

        self.layout_unit_count = unit_count 


    def apply_map_layout(self, i):
        self.reset_map()
        layout = self.map_layouts[i]

        for i in range(self.layout_unit_count):
            if (i/self.layout_unit_count) <= 1/8 or (i/self.layout_unit_count) >= 6/8:
                unit_type = 'cavalry'
            elif (i/self.layout_unit_count) <= 2/8 or (i/self.layout_unit_count) >= 5/8: 
                unit_type = 'archer'
            else:
                unit_type = 'infantry'
            self.place_unit(layout[i], unit_type, 0)
        for i in range(self.layout_unit_count):
            if (i/self.layout_unit_count) <= 1/8 or (i/self.layout_unit_count) >= 6/8:
                unit_type = 'cavalry'
            elif (i/self.layout_unit_count) <= 2/8 or (i/self.layout_unit_count) >= 5/8: 
                unit_type = 'archer'
            else:
                unit_type = 'infantry'
            self.place_unit(layout[self.layout_unit_count+i], unit_type, 1)  

    def copy_setup(self, other_manager):
        self.dimensions = other_manager.Map.shape
        self.layout_unit_count = other_manager.layout_unit_count 
        self.map_layouts = copy.deepcopy(other_manager.map_layouts)


    def game_joever(self):
        if len(self.Teams[0].live_units) == 0:
            return 1
        if len(self.Teams[1].live_units) == 0:
            return 0
        return -1

    #returns -1 if the game is not over, else returns the number of the winning team
    def game_feedback(self):
        #No units
        # if len(self.Teams[0].live_units) == 0:
        #     return 1
        # if len(self.Teams[1].live_units) == 0:
        #     return 0

        #More units
        if len(self.Teams[0].live_units) > len(self.Teams[1].live_units):
            return 0
        if len(self.Teams[1].live_units) > len(self.Teams[0].live_units):
            return 1
        #return -1

        #Units difference
        #return len(self.Teams[0].live_units) - len(self.Teams[1].live_units)
    
class Tile:    
    class Terrain:
        def __init__(self, name, range_mod, def_mod, tile_cost):
            self.name = name
            self.range_mod = range_mod
            self.def_mod = def_mod
            self.tile_cost = tile_cost

    terrain_types = {
        'Plains': Terrain('Plains', 0, 1.0, 1),
        'Hills': Terrain('Hills', 1, 2.0, 2),
        'Forest': Terrain('Forest', -2, 0.5, 3), 
    }

    def __init__(self, pos, unit, terrain_name):
        self.pos = pos
        self.unit_ref = unit
        self.terrain = Tile.terrain_types[terrain_name]
        #Cost to move onto this tile
        self.tile_cost = self.terrain.tile_cost

        self.rotation = -1
        #Cost of TOTAL path to this tile
        self.move_cost = float('inf')
        self.move_parent = None
        self.visited = False
        self.is_attack = False

    def __str__(self):
        string = ''
        if (self.visited):
            if self.is_attack:
                string += Back.RED
            else:
                string += Back.BLUE
        if (self.terrain.name == 'Hills'):
            string += Fore.YELLOW
        if (self.terrain.name == 'Forest'):
            string += Fore.GREEN
        endstring = Style.RESET_ALL
        if (self.unit_ref == None):
            return string + "[ ]" + endstring
        else:
            return string + "[{}]".format(self.unit_ref.map_char) + endstring
        
    def __lt__(self, other):
        return self.move_cost < other.move_cost
    
    def print_tile(self):
        return "Tile ({0}, {1}):\nUnit: {2}".format(self.pos[0], self.pos[1], self.unit_ref)
        
class Team:
    def __init__(self, num):
        self.team_num = num
        self.units = []
        self.live_units = []

class Unit:
    unit_rotations = {'E': 0, 'S': 1, 'W': 2, 'N':3} 
    #Note: positive 1 in y direction == going downwards
    move_to_rotation = {(0, 1) : 0, (1, 0) : 1, (0, -1) : 2, (-1, 0) : 3} 

    #Default melee range is 0
    unit_types = {
        'archer':{'att':20, 'defs':5, 'range':2, 'move':2, 'char':'a'},
        'infantry':{'att':20, 'defs':10, 'range':0, 'move':2, 'char':'i'},
        'cavalry':{'att':30, 'defs':5, 'range':0, 'move':4, 'char':'c'}
    }

    def is_flank(def_unit, att_unit):
        if def_unit.rotation == att_unit.rotation:
            return True
        return False


    def __init__(self, pos, unit_type, Team):
        self.pos = pos
        self.hp = 100
        self.temp_hp = self.hp
        self.type = unit_type
        
        self.max_move = Unit.unit_types[self.type]['move']
        self.curr_move = self.max_move
        self.range = Unit.unit_types[self.type]['range'] 
        self.Att = Unit.unit_types[self.type]['att'] 
        self.Def = Unit.unit_types[self.type]['defs'] 
        self.map_char = Unit.unit_types[self.type]['char']

        self.Team = Team
        if self.Team == 0:
            self.rotation = Unit.unit_rotations['E']
        elif self.Team == 1:
            self.map_char = self.map_char.upper()
            self.rotation = Unit.unit_rotations['W']
        
    def __str__(self):
        string = "Unit:\n\tpos: {0}\n\tcurr_move: {1}\n\thp: {2}\n".format(self.pos, self.curr_move, self.hp) 
        string +="\ttemp_hp: {0}\n\tType: {1}\n".format(self.temp_hp, self.type)
        string += "\tmax_move: {0}\n\tAtt: {1}\n\tDef: {2}\n\tTeam: {3}\n".format(self.max_move, self.Att, self.Def, self.Team)
        string +="\trotation: {0}\n".format(self.rotation)
        return string