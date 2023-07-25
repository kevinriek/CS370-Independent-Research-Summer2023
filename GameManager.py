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
                Map[i,j] = Tile((i, j), None)
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
                    if (Map[new_pos].move_cost > new_cost and new_cost <= unit.max_move):
                        Map[new_pos].move_cost = new_cost
                        Map[new_pos].move_parent = curr_tile
                        if (Map[new_pos].unit_ref is not None):
                            Map[new_pos].is_attack = True
                            Map[new_pos].visited = True
                        else:
                            #Should prevent movement through opponents
                            prio_queue.put(Map[new_pos])
                        self.visited_tiles.append(Map[new_pos])
                        
                    
        
    def place_unit(self, pos, team):
        new_unit = Unit(pos=pos, hp=100, mmove=3, Att=20, Def=10, Team=team)
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
        self.visited_tiles = []
        
    
    def move_unit(self, unit, pos):
        #move_pos = tuple(np.add(unit.pos, dir))
        if pos == unit.pos:
            return
        
        move_pos = pos
        tile = self.Map[move_pos]
        """        try:                 #CURRENTLY AM NOT SETTING THE VISITED STAT SO THIS DOES NOT WORK
            assert(tile.visited)
        except AssertionError:
            print("Tile {0} not reachable by unit".format(move_pos))"""

        if(tile.is_attack == True):
            move_tile = tile.move_parent 
            self.Map[unit.pos].unit_ref = None
            unit.pos = move_tile.pos
            unit.curr_move -= move_tile.move_cost
            move_tile.unit_ref = unit
            
            target_unit = tile.unit_ref
            self.combat(unit, tile.unit_ref)
        else:
            move_tile = tile
            self.Map[unit.pos].unit_ref = None
            unit.pos = move_tile.pos
            unit.curr_move -= move_tile.move_cost
            move_tile.unit_ref = unit
        
    def sim_combat(self, att_unit, def_unit):
        #att_dmg = (def_unit.Def / att_unit.Att) * 20
        def_dmg = (att_unit.Att / def_unit.Def) * 25

        def_unit.temp_hp = def_unit.hp - def_dmg
        if (def_unit.temp_hp <= 0):
            def_unit.temp_hp = 0

    def reset_temp_hp(self):
        for unit in self.Units:
            unit.temp_hp = unit.hp

    def combat(self, att_unit, def_unit):
        #att_dmg = (def_unit.Def / att_unit.Att) * 20
        def_dmg = (att_unit.Att / def_unit.Def) * 25
        #att_unit.hp -= att_dmg
        def_unit.hp -= def_dmg
        def_unit.temp_hp = def_unit.hp

        if (def_unit.hp <= 0):
            def_unit.hp = 0
            def_unit.temp_hp = def_unit.hp
            
            #self.Units.remove(def_unit)   #don't remove from all units, this messes up inputs
            self.Teams[def_unit.Team].live_units.remove(def_unit)
            self.Map[def_unit.pos].unit_ref = None
            #del def_unit
    

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
            self.place_unit(layout[i], 0)
        for i in range(self.layout_unit_count):
            self.place_unit(layout[self.layout_unit_count+i], 1)  


    def setup_rand(self, unit_count):
        dimensions = self.Map.shape
        self.reset_map()
        
        pos_ls = []
        for i in range(unit_count):
            pos_ls.append((round(random.uniform((dimensions[0]/unit_count)*i, (dimensions[1]/unit_count)*(i+1)-1)),
                            round(random.uniform(0, dimensions[1]/7))))
        for i in range(unit_count):
            pos_ls.append((round(random.uniform((dimensions[0]/unit_count)*i, (dimensions[1]/unit_count)*(i+1)-1)),
                            round(random.uniform(dimensions[1]-(1+dimensions[1]/7), dimensions[1]-1))))

        team1 = 0
        team2 = 1

        for i in range(unit_count):
            self.place_unit(pos_ls[i], team1)
        for i in range(unit_count):
            self.place_unit(pos_ls[unit_count+i], team2)   
        

    def setup_even(self, unit_count):
        dimensions = self.Map.shape
        pos_pick = random.randint(0, 1)
        pos_list = []

        start_offset = (dimensions[0] - unit_count) // 2
        for i in range(unit_count):
            pos_list.append((i+start_offset, 0))
        for i in range(unit_count):
            pos_list.append((i+start_offset, dimensions[1]-1))

        self.reset_map()
        if pos_pick == 0:
            for i in range(unit_count):
                self.place_unit(pos_list[i], 0)
            for i in range(unit_count, unit_count*2):
                self.place_unit(pos_list[i], 1)
        else:
            for i in range(unit_count):
                self.place_unit(pos_list[i], 1)
            for i in range(unit_count, unit_count*2):
                self.place_unit(pos_list[i], 0)

    def game_joever(self):
        if len(self.Teams[0].live_units) == 0:
            return 1
        if len(self.Teams[1].live_units) == 0:
            return 0
        return -1

    #returns -1 if the game is not over, else returns the number of the winning team
    def game_feedback(self):
        #No units
        if len(self.Teams[0].live_units) == 0:
            return 1
        if len(self.Teams[1].live_units) == 0:
            return 0

        #More units
        # if len(self.Teams[0].live_units) > len(self.Teams[1].live_units):
        #     return 0
        # if len(self.Teams[1].live_units) > len(self.Teams[0].live_units):
        #     return 1
        # return -1

        #Units difference
        #return len(self.Teams[0].live_units) - len(self.Teams[1].live_units)
    
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
        self.live_units = []

class Unit:
    def __init__(self, pos, hp, mmove, Att, Def, Team):
        self.pos = pos
        self.hp = hp
        self.temp_hp = hp
        
        self.max_move = mmove
        self.curr_move = mmove

        self.Att = Att
        self.Def = Def
        self.Team = Team
        
    def __str__(self):
        string = "Unit:\n\tpos: {0}\n\tcurr_move: {1}\n\thp: {2}\n".format(self.pos, self.curr_move, self.hp) 
        string +="\ttemp_hp: {0}\n".format(self.temp_hp)
        string += "\tmax_move: {0}\n\tAtt: {1}\n\tDef: {2}\n\tTeam: {3}\n".format(self.max_move, self.Att, self.Def, self.Team)
        return string