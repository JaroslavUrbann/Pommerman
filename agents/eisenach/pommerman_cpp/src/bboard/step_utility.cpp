#include <iostream>

#include "bboard.hpp"
#include "step_utility.hpp"

namespace bboard::util
{

Position DesiredPosition(int x, int y, Move move)
{
    Position p;
    p.x = x;
    p.y = y;
    if(move == Move::UP)
    {
        p.y -= 1;
    }
    else if(move == Move::DOWN)
    {
        p.y += 1;
    }
    else if(move == Move::LEFT)
    {
        p.x -= 1;
    }
    else if(move == Move::RIGHT)
    {
        p.x += 1;
    }
    return p;
}
void FillDestPos(State* s, Move m[AGENT_COUNT], Position p[AGENT_COUNT])
{
    for(int i = 0; i < AGENT_COUNT; i++)
    {
        p[i] = DesiredPosition(s->agents[i].x, s->agents[i].y, m[i]);
    }
}

bool FixSwitchMove(State* s, Position d[AGENT_COUNT])
{
    bool any_switch = false;
    //If they want to step each other's place, nobody goes anywhere
    for(int i = 0; i < AGENT_COUNT; i++)
    {
        if(s->agents[i].dead || s->agents[i].x < 0)
        {
            continue;
        }
        for(int j = i + 1; j < AGENT_COUNT; j++)
        {

            if(s->agents[j].dead || s->agents[j].x < 0)
            {
                continue;
            }
            if(d[i].x == s->agents[j].x && d[i].y == s->agents[j].y &&
                    d[j].x == s->agents[i].x && d[j].y == s->agents[i].y)
            {
                any_switch = true;
                d[i].x = s->agents[i].x;
                d[i].y = s->agents[i].y;
                d[j].x = s->agents[j].x;
                d[j].y = s->agents[j].y;
            }
        }
    }
    return any_switch;
}

void MoveBombs(State* state, Position d[AGENT_COUNT])
{
    for(int bombIndex=0; bombIndex <state->bombs.count; bombIndex++) {
        if (BMB_VEL(state->bombs[bombIndex]) > 0) {
            Position desiredPos = bboard::util::DesiredPosition(BMB_POS_X(state->bombs[bombIndex]), BMB_POS_Y(state->bombs[bombIndex]),
                                                                (bboard::Move) BMB_VEL(state->bombs[bombIndex]));
            if (bboard::_CheckPos_any(state, desiredPos.x, desiredPos.y)) {
                bool agentWantsToMoveThere = false;
                for (int i = 0; i < AGENT_COUNT; i++) {
                    if(!state->agents[i].dead && state->agents[i].x >=0 && d[i].x == desiredPos.x && d[i].y == desiredPos.y) {
                        //Agent stays
                        d[i].x = state->agents[i].x;
                        d[i].y = state->agents[i].y;
                        SetBombVelocity(state->bombs[bombIndex], 0);
                        break;
                    }
                }
                if (!agentWantsToMoveThere) {
                    bool explodes = IS_FLAME(state->board[desiredPos.y][desiredPos.x]);
                    state->board[BMB_POS_Y(state->bombs[bombIndex])][BMB_POS_X(state->bombs[bombIndex])] = PASSAGE;
                    SetBombPosition(state->bombs[bombIndex], desiredPos.x, desiredPos.y);
                    state->board[desiredPos.y][desiredPos.x] = BOMB;
                    if (explodes) {
                        state->ExplodeBomb(bombIndex);
                        bombIndex--;
                    }
                }
            } else {
                SetBombVelocity(state->bombs[bombIndex], 0);
            }
        }
    }
}

int ResolveDependencies(State* s, Position des[AGENT_COUNT],
                        int dependency[AGENT_COUNT], int chain[AGENT_COUNT])
{
    int rootCount = 0;
    for(int i = 0; i < AGENT_COUNT; i++)
    {
        // dead agents are handled as roots
        // also invisible agents
        if(s->agents[i].dead || s->agents[i].x < 0)
        {
            chain[rootCount] = i;
            rootCount++;
            continue;
        }

        bool isChainRoot = true;
        for(int j = 0; j < AGENT_COUNT; j++)
        {
            if(i == j || s->agents[j].dead || s->agents[j].x < 0) continue;

            if(des[i].x == s->agents[j].x && des[i].y == s->agents[j].y)
            {
                if(dependency[j] == -1) {
                    dependency[j] = i;
                }else{
                    dependency[dependency[j]] = i;
                }
                isChainRoot = false;
                break;
            }
        }
        if(isChainRoot)
        {
            chain[rootCount] = i;
            rootCount++;
        }
    }
    return rootCount;
}


void TickFlames(State& state)
{
    for(int i = 0; i < state.flames.count; i++)
    {
        state.flames[i].timeLeft--;
    }
    int flameCount = state.flames.count;
    for(int i = 0; i < flameCount; i++)
    {
        if(state.flames[0].timeLeft == 0)
        {
            state.PopFlame();
        }
    }
}

void TickBombs(State& state)
{
    for(int i = 0; i < state.bombs.count; i++)
    {
        ReduceBombTimer(state.bombs[i]);
    }
    //explode timed-out bombs
    for(int i = 0; i < state.bombs.count; i++)
    {
        if(BMB_TIME(state.bombs[0]) == 0)
        {
            __glibcxx_assert(i == 0);
            state.ExplodeTopBomb();
            i--;
        }else{
            break;
        }
    }
}


    void TickAndMoveBombs10(State& state){
        //explode timed-out bombs
        int stepSize = 1;
        for(int remainingTime = 10; remainingTime>0; remainingTime-=stepSize) {
            __glibcxx_assert(stepSize > 0);
            bool anyBodyMoves = false;
            int minRemaining = 100;
            state.relTimeStep += stepSize;
            for (int i = 0; i < state.bombs.count; i++) {
                for(int t = 0; t<stepSize; t++) //TODO: shift in one turn?
                    ReduceBombTimer(state.bombs[i]);

                if (BMB_TIME(state.bombs[i]) == 0) {
                    __glibcxx_assert(i == 0);
                    state.ExplodeTopBomb();
                    i--;
                } else {
                    minRemaining = std::min(minRemaining, BMB_TIME(state.bombs[i]));
                    if (BMB_VEL(state.bombs[i]) > 0) {
                        Position desiredPos = bboard::util::DesiredPosition(BMB_POS_X(state.bombs[i]), BMB_POS_Y(state.bombs[i]),
                                                                            (bboard::Move) BMB_VEL(state.bombs[i]));
                        if (_CheckPos_any(&state, desiredPos.x, desiredPos.y)) {
                            //TODO: Add if(explodes)... from other locations function is used
                            state.board[BMB_POS_Y(state.bombs[i])][BMB_POS_X(state.bombs[i])] = PASSAGE;
                            SetBombPosition(state.bombs[i], desiredPos.x, desiredPos.y);
                            state.board[desiredPos.y][desiredPos.x] = BOMB;
                            anyBodyMoves = true;
                        } else {
                            SetBombVelocity(state.bombs[i], 0);
                        }
                    }
                }
            }

            stepSize = (anyBodyMoves ? 1 : minRemaining);
            //Exit if match decided, maybe we would die later from an other bomb, so that disturbs pointing and decision making
            if (state.aliveAgents < 2 || (state.aliveAgents == 2 && ((state.agents[0].dead && state.agents[2].dead) || (state.agents[1].dead && state.agents[3].dead))))
                break;
        }
    }

void ConsumePowerup(State& state, int agentID, int powerUp)
{
    if(powerUp == Item::EXTRABOMB)
    {
        state.agents[agentID].maxBombCount++;
        state.agents[agentID].extraBombPowerupPoints += 1.0 - state.relTimeStep/100.0;
    }
    else if(powerUp == Item::INCRRANGE)
    {
        state.agents[agentID].bombStrength++;
        state.agents[agentID].extraRangePowerupPoints += 1.0 - state.relTimeStep/100.0;
    }
    else if(powerUp == Item::KICK)
    {
        if(state.agents[agentID].canKick)
        {
            state.agents[agentID].otherKickPowerupPoints += 1.0 - state.relTimeStep/100.0;
        }else{
            state.agents[agentID].firstKickPowerupPoints += 1.0 - state.relTimeStep/100.0;
        }
        state.agents[agentID].canKick = true;
    }

}

bool HasDPCollision(const State& state, Position dp[AGENT_COUNT], int agentID)
{
    for(int i = 0; i < AGENT_COUNT; i++)
    {
        if(agentID == i || state.agents[i].dead || state.agents[i].x < 0) continue;
        if(dp[agentID] == dp[i])
        {
            // a destination position conflict will never
            // result in a valid move
            return true;
        }
    }
    return false;
}

void PrintDependency(int dependency[AGENT_COUNT])
{
    for(int i = 0; i < AGENT_COUNT; i++)
    {
        if(dependency[i] == -1)
        {
            std::cout << "[" << i << " <- ]";
        }
        else
        {
            std::cout << "[" << i << " <- " << dependency[i] << "]";
        }
        std::cout << std::endl;
    }
}

void PrintDependencyChain(int dependency[AGENT_COUNT], int chain[AGENT_COUNT])
{
    for(int i = 0; i < AGENT_COUNT; i++)
    {
        if(chain[i] == -1) continue;

        std::cout << chain[i];
        int k = dependency[chain[i]];

        while(k != -1)
        {
            std::cout << " <- " << k;
            k = dependency[k];
        }
        std::cout << std::endl;
    }
}


}
