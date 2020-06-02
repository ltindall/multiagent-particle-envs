import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2 # communication dimension set but communication info never observed
        num_trapped_agents = 1
        num_rescue_agents = 3
        num_adversary_agents = 3
        num_agents = num_adversary_agents + num_rescue_agents + num_trapped_agents
        num_landmarks = 2
        world.agents = []
        
        # add trapped agents
        for i in range(num_trapped_agents): 
            agent = Agent()
            agent.name = 'trapped_agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.agent_type = 'trapped'
            agent.adversary = False
            #agent.trapped = True
            agent.size =  0.05
            agent.accel =  4.0
            agent.max_speed = 1.3
            agent.color = np.array([0.35, 0.85, 0.35]) # green 
            world.agents.append(agent)
        
        # add rescue agents 
        for i in range(num_rescue_agents): 
            agent = Agent()
            agent.name = 'rescue_agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.agent_type = 'rescue'
            agent.adversary = False
            #agent.trapped = False 
            agent.size =  0.05
            agent.accel =  4.0
            agent.max_speed = 1.3
            agent.color = np.array([0.35, 0.35, 0.85]) # blue
            world.agents.append(agent)

        # add adversary agents 
        for i in range(num_adversary_agents): 
            agent = Agent()
            agent.name = 'adversary_agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.agent_type = 'adversary'
            agent.adversary = True
            agent.size =  0.075
            agent.accel =  3.0
            agent.max_speed = 1.0
            agent.color = np.array([0.85, 0.35, 0.35]) # red 
            world.agents.append(agent)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark_%d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
            landmark.color = np.array([0.25, 0.25, 0.25]) # grey 
        
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.agent_type == 'adversary':
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def trapped_agents(self, world):
        return [agent for agent in world.agents if agent.agent_type == 'trapped']

    # return all agents that are not adversaries
    def rescue_agents(self, world):
        return [agent for agent in world.agents if agent.agent_type == 'rescue']

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.agent_type == 'adversary']


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.agent_type == 'adversary' else self.agent_reward(agent, world)
        return main_reward


    ### TODO: REWARD FOR RESCUERS IS WRONG, IT PENALIZES COLLISIONS WHICH WOULD BE A GOOD STRATEGY FOR DEFENDERS 
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        trapped_agents = self.trapped_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in trapped_agents])
        if agent.collide:
            for ag in trapped_agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if other.agent_type != 'adversary':
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
