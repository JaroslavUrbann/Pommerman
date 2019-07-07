#include <thread>
#include <chrono>
#include <functional>

#include "bboard.hpp"
#include "step_utility.hpp"

namespace bboard
{

void Pause(bool timeBased)
{
    if(!timeBased)
    {
        std::cin.get();
    }
    else
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void PrintGameResult(Environment& env)
{
    std::cout << std::endl;

    if(env.IsDone())
    {
        if(env.IsDraw())
        {
            std::cout << "Draw! All agents are dead"
                      << std::endl;
        }
        else
        {
            std::cout << "Finished! The winner is Agent "
                      << env.GetWinner() << std::endl;
        }

    }
    else
    {
        std::cout << "Draw! Max timesteps reached "
                  << std::endl;
    }
}

Environment::Environment()
{
    state = std::make_unique<State>();
}

void Environment::MakeGame(std::array<Agent*, AGENT_COUNT> a)
{
    bboard::InitState(state.get(), 0, 1, 2, 3);

    state->PutAgentsInCorners(0, 1, 2, 3);

    SetAgents(a);
    hasStarted = true;
}

void Environment::MakeGameFromPython(int ourId)
{
    state->ourId = ourId;
    for(int x=0; x<11; x++)
        for(int y=0; y<11; y++)
            state->board[y][x] = FOG;
}

    void Environment::MakeGameFromPython_berlin(bool agent1Alive, bool agent2Alive, bool agent3Alive, uint8_t * board, double * bomb_life,
                                                double * bomb_blast_strength, int posx, int posy, int blast_strength, bool can_kick, int ammo, int teammate_id)
    {
        state->bombs.count = 0;
        //state->flames.count = 0; //keep!
        state->agents[0].x = -1;
        state->agents[0].y = -1;
        state->agents[1].x = -1;
        state->agents[1].y = -1;
        state->agents[2].x = -1;
        state->agents[2].y = -1;
        state->agents[3].x = -1;
        state->agents[3].y = -1;

        state->agents[0].extraBombPowerupPoints = 0;
        state->agents[0].extraRangePowerupPoints = 0;
        state->agents[0].firstKickPowerupPoints = 0;
        state->agents[0].otherKickPowerupPoints = 0;
        state->agents[1].extraBombPowerupPoints = 0;
        state->agents[1].extraRangePowerupPoints = 0;
        state->agents[1].firstKickPowerupPoints = 0;
        state->agents[1].otherKickPowerupPoints = 0;
        state->agents[2].extraBombPowerupPoints = 0;
        state->agents[2].extraRangePowerupPoints = 0;
        state->agents[2].firstKickPowerupPoints = 0;
        state->agents[2].otherKickPowerupPoints = 0;
        state->agents[3].extraBombPowerupPoints = 0;
        state->agents[3].extraRangePowerupPoints = 0;
        state->agents[3].firstKickPowerupPoints = 0;
        state->agents[3].otherKickPowerupPoints = 0;
        state->agents[0].woodDemolished = 0;
        state->agents[1].woodDemolished = 0;
        state->agents[2].woodDemolished = 0;
        state->agents[3].woodDemolished = 0;
        state->timeStep++;
        state->teammateId = teammate_id - 10;

        if(state->ourId == 1 || state->ourId == 3)
        {
            state->enemy1Id = 0;
            state->enemy2Id = 2;
        }else
        {
            state->enemy1Id = 1;
            state->enemy2Id = 3;
        }

        //std::cout << state->ourId << " " << state->teammateId << " : " << state->enemy1Id << " " << state->enemy2Id << std::endl;

        for(int i=0; i<11*11; i++)
        {
            int y = i/11;
            int x = i%11;
            switch(board[i])
            {
                case PyPASSAGE:
                    state->board[y][x] = PASSAGE;
                    break;
                case PyRIGID:
                    state->board[y][x] = RIGID;
                    break;
                case PyWOOD:
                    state->board[y][x] = WOOD;
                    break;
                case PyBOMB:
                    state->board[y][x] = BOMB;
                    {
                        int bombId = state->ourId;//TODO: should be the agent who placed it
                        Bomb* b = &state->bombs.NextPos();
                        SetBombID(*b, bombId);
                        SetBombPosition(*b, x, y);
                        SetBombStrength(*b, bomb_blast_strength[i] - 1);
                        // TODO: velocity
                        SetBombTime(*b, bomb_life[i]);
                        state->agents[bombId].bombCount++;
                        state->bombs.count++;
                    }

                    break;
                case PyFLAMES:
                    state->board[y][x] = FLAMES;
                    break;
                case PyFOG:
                    state->board[y][x] = FOG;
                    break;
                case PyEXTRABOMB:
                    state->board[y][x] = EXTRABOMB;
                    break;
                case PyINCRRANGE:
                    state->board[y][x] = INCRRANGE;
                    break;
                case PyKICK:
                    state->board[y][x] = KICK;
                    break;
                case PyAGENTDUMMY:
                    state->board[y][x] = AGENTDUMMY;
                    break;
                case PyAGENT0:
                case PyAGENT1:
                case PyAGENT2:
                case PyAGENT3:
                    state->board[y][x] = AGENT0 + (board[i] - PyAGENT0);
                    state->agents[board[i] - PyAGENT0].x = x;
                    state->agents[board[i] - PyAGENT0].y = y;
                    if(bomb_blast_strength[i] > 0) //hidden bomb under us
                    {
                        int bombId = state->ourId;//TODO: should be the agent who placed it
                        Bomb* b = &state->bombs.NextPos();
                        SetBombID(*b, bombId);
                        SetBombPosition(*b, x, y);
                        SetBombStrength(*b, bomb_blast_strength[i] - 1);
                        // TODO: velocity
                        SetBombTime(*b, bomb_life[i]);
                        state->agents[bombId].bombCount++;
                        state->bombs.count++;
                        //std::cout << "Bomb under agent " << board[i] - PyAGENT0 << " at " << x << " " << y << " time: " << bomb_life[i] << std::endl;
                    }
                    break;
                default:
                    std::cout << "Unknown item in py board: " << board[i] << std::endl;
            }
        }

        state->aliveAgents = 1 + int(agent1Alive) + int(agent2Alive) + int(agent3Alive);
        state->agents[state->ourId].x = posy;
        state->agents[state->ourId].y = posx;
        state->agents[state->ourId].canKick = can_kick;
        state->agents[state->ourId].bombCount = state->agents[state->ourId].maxBombCount - ammo;
        state->agents[state->ourId].bombStrength = blast_strength - 1;

        int i, j;
        for (i = 0; i < state->bombs.count-1; i++)
            for (j = 0; j < state->bombs.count-1-i; j++)
                if (BMB_TIME(state->bombs[j]) > BMB_TIME(state->bombs[j+1]))
                    std::swap(state->bombs[j], state->bombs[j+1]);
    }

    void Environment::MakeGameFromPython_cologne(bool agent0Alive, bool agent1Alive, bool agent2Alive, bool agent3Alive, uint8_t * board, double * bomb_life,
                                                 double * bomb_blast_strength, int posx, int posy, int blast_strength, bool can_kick, int ammo, int teammate_id)
    {
        state->agents[0].bombCount = 0;
        state->agents[1].bombCount = 0;
        state->agents[2].bombCount = 0;
        state->agents[3].bombCount = 0;
        state->agents[0].diedAt = state->timeStep;
        state->agents[1].diedAt = state->timeStep;
        state->agents[2].diedAt = state->timeStep;
        state->agents[3].diedAt = state->timeStep;
        state->agents[0].dead = !agent0Alive;
        state->agents[1].dead = !agent1Alive;
        state->agents[2].dead = !agent2Alive;
        state->agents[3].dead = !agent3Alive;

        for(int i=0; i < state->bombs.count; i++)
        {
            int boardPos = BMB_POS_X(state->bombs[i]) + BMB_POS_Y(state->bombs[i]) * 11;
            if(board[boardPos] == PyPASSAGE || board[boardPos] == PyFLAMES || BMB_TIME(state->bombs[i]) == 1) {
                state->bombs.RemoveAt(i);
                i--;
            }else {
                ReduceBombTimer(state->bombs[i]); //Can be read from bomb_life array, if not in fog
                state->agents[BMB_ID(state->bombs[i])].bombCount++;
            }
        }
        for(int i=0; i < state->powerup_incr.count; i++)
        {
            if(board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] >= PyAGENT0)
                state->agents[board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] - PyAGENT0].bombStrength++;
            else if(board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] == PyFOG)
                board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] = PyINCRRANGE; //so it will be added again
        }
        for(int i=0; i < state->powerup_extrabomb.count; i++)
        {
            if(board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] >= PyAGENT0)
                state->agents[board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] - PyAGENT0].maxBombCount++;
            else if(board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] == PyFOG)
                board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] = PyEXTRABOMB; //so it will be added again
        }
        for(int i=0; i < state->powerup_kick.count; i++)
        {
            if(board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] >= PyAGENT0)
                state->agents[board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] - PyAGENT0].canKick = true;
            else if(board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] == PyFOG)
                board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] = PyKICK; //so it will be added again
        }
        for(int i=0; i < state->woods.count; i++)
        {
            if(board[state->woods[i].x + state->woods[i].y * 11] == PyFOG)
                board[state->woods[i].x + state->woods[i].y * 11] = PyWOOD; //so it will be added again
        }

        state->woods.count = 0;
        state->powerup_incr.count = 0;
        state->powerup_kick.count = 0;
        state->powerup_extrabomb.count = 0;
        //state->flames.count = 0; //keep!
        state->agents[0].x = -1;
        state->agents[0].y = -1;
        state->agents[1].x = -1;
        state->agents[1].y = -1;
        state->agents[2].x = -1;
        state->agents[2].y = -1;
        state->agents[3].x = -1;
        state->agents[3].y = -1;

        state->agents[0].extraBombPowerupPoints = 0;
        state->agents[0].extraRangePowerupPoints = 0;
        state->agents[0].firstKickPowerupPoints = 0;
        state->agents[0].otherKickPowerupPoints = 0;
        state->agents[1].extraBombPowerupPoints = 0;
        state->agents[1].extraRangePowerupPoints = 0;
        state->agents[1].firstKickPowerupPoints = 0;
        state->agents[1].otherKickPowerupPoints = 0;
        state->agents[2].extraBombPowerupPoints = 0;
        state->agents[2].extraRangePowerupPoints = 0;
        state->agents[2].firstKickPowerupPoints = 0;
        state->agents[2].otherKickPowerupPoints = 0;
        state->agents[3].extraBombPowerupPoints = 0;
        state->agents[3].extraRangePowerupPoints = 0;
        state->agents[3].firstKickPowerupPoints = 0;
        state->agents[3].otherKickPowerupPoints = 0;
        state->agents[0].woodDemolished = 0;
        state->agents[1].woodDemolished = 0;
        state->agents[2].woodDemolished = 0;
        state->agents[3].woodDemolished = 0;
        state->timeStep++;
        state->teammateId = teammate_id - 10;

        if(state->ourId == 1 || state->ourId == 3)
        {
            state->enemy1Id = 0;
            state->enemy2Id = 2;
        }else
        {
            state->enemy1Id = 1;
            state->enemy2Id = 3;
        }

        //std::cout << state->ourId << " " << state->teammateId << " : " << state->enemy1Id << " " << state->enemy2Id << std::endl;

        for(int i=0; i<11*11; i++)
        {
            int y = i/11;
            int x = i%11;
            switch(board[i])
            {
                case PyPASSAGE:
                    state->board[y][x] = PASSAGE;
                    break;
                case PyRIGID:
                    state->board[y][x] = RIGID;
                    break;
                case PyWOOD:
                    state->board[y][x] = WOOD;
                    {Position p; p.x = x; p.y = y; state->woods.NextPos() = p; state->woods.count++;}
                    break;
                case PyBOMB:
                    //Already known bombs are handled, hopefully for them:
                    //  BMB_TIME(state->bombs[?]) == bomb_life[i]
                    //  BMB_STRENGTH(state->bombs[?]) == bomb_blast_strength[i] - 1
                    //New bombs are born under an agent, and added there. so here in theory we only handle bombs which were placed in fog and discovered later
                    state->board[y][x] = BOMB;
                    {
                        bool already_handled = false;
                        for (int bombindex = 0; bombindex < state->bombs.count; bombindex++) {
                            if(BMB_POS_X(state->bombs[bombindex]) == x && BMB_POS_Y(state->bombs[bombindex]) == y)
                            {
                                already_handled = true;
                                SetBombTime(state->bombs[bombindex], bomb_life[i]); //should already be it !!
                                break;
                            }
                        }
                        if(!already_handled)
                        {
                            int bombId = state->ourId; //The owner is unknown :(
                            Bomb *b = &state->bombs.NextPos();
                            SetBombID(*b, bombId);
                            SetBombPosition(*b, x, y);
                            SetBombStrength(*b, (int)bomb_blast_strength[i] - 1);
                            // TODO: velocity
                            SetBombTime(*b, bomb_life[i]);
                            state->agents[bombId].bombCount++;
                            state->bombs.count++;
                            std::cout << "Found an unknown bomb at " << y << x << std::endl;
                        }
                    }
                    break;
                case PyFLAMES:
                    state->board[y][x] = FLAMES;
                    break;
                case PyFOG:
                    if(state->board[y][x] >= AGENT0)
                        state->board[y][x] = FOG;
                    /*if(state->board[y][x] == RIGID)
                        state->board[y][x] = RIGID;
                    else
                        state->board[y][x] = FOG;*/
                    break;
                case PyEXTRABOMB:
                    state->board[y][x] = EXTRABOMB;
                    {Position p; p.x = x; p.y = y; state->powerup_extrabomb.NextPos() = p; state->powerup_extrabomb.count++;}
                    break;
                case PyINCRRANGE:
                    state->board[y][x] = INCRRANGE;
                    {Position p; p.x = x; p.y = y; state->powerup_incr.NextPos() = p; state->powerup_incr.count++;}
                    break;
                case PyKICK:
                    state->board[y][x] = KICK;
                    {Position p; p.x = x; p.y = y; state->powerup_kick.NextPos() = p; state->powerup_kick.count++;}
                    break;
                case PyAGENTDUMMY:
                    state->board[y][x] = AGENTDUMMY;
                    break;
                case PyAGENT0:
                case PyAGENT1:
                case PyAGENT2:
                case PyAGENT3:
                    state->board[y][x] = AGENT0 + (board[i] - PyAGENT0);
                    state->agents[board[i] - PyAGENT0].x = x;
                    state->agents[board[i] - PyAGENT0].y = y;
                    if((int)bomb_blast_strength[i] > 0) //hidden bomb under us
                    {
                        bool already_handled = false;
                        for (int bombindex = 0; bombindex < state->bombs.count; bombindex++) {
                            if(BMB_POS_X(state->bombs[bombindex]) == x && BMB_POS_Y(state->bombs[bombindex]) == y)
                            {
                                already_handled = true;
                                SetBombTime(state->bombs[bombindex], bomb_life[i]); //should already be it !!
                                break;
                            }
                        }
                        if(!already_handled) {
                            int bombId = board[i] - PyAGENT0;//the agent who placed it
                            Bomb *b = &state->bombs.NextPos();
                            SetBombID(*b, bombId);
                            SetBombPosition(*b, x, y);
                            SetBombStrength(*b, (int)bomb_blast_strength[i] - 1);
                            // TODO: velocity
                            SetBombTime(*b, bomb_life[i]);
                            if (bomb_life[i] > 8) //Let's assume it was placed by the guy on the top
                            {
                                //extra info: the agent's bomb-strength
                                state->agents[bombId].bombStrength = (int)bomb_blast_strength[i] - 1;
                            }
                            state->agents[bombId].bombCount++;
                            state->bombs.count++;
                        }
                        std::cout << "Bomb under agent " << board[i] - PyAGENT0 << " at " << x << " " << y << " time: " << bomb_life[i] << std::endl;
                    }
                    break;
                default:
                    std::cout << "Unknown item in py board: " << board[i] << std::endl;
            }
        }

        state->agents[state->enemy1Id].maxBombCount = std::max(state->agents[state->enemy1Id].maxBombCount, state->agents[state->enemy1Id].bombCount);
        state->agents[state->enemy2Id].maxBombCount = std::max(state->agents[state->enemy2Id].maxBombCount, state->agents[state->enemy2Id].bombCount);
        state->agents[state->teammateId].maxBombCount = std::max(state->agents[state->teammateId].maxBombCount, state->agents[state->teammateId].bombCount);

        state->aliveAgents = int(agent0Alive) + int(agent1Alive) + int(agent2Alive) + int(agent3Alive);
        state->agents[state->ourId].x = posy;
        state->agents[state->ourId].y = posx;
        state->agents[state->ourId].canKick = can_kick;
        state->agents[state->ourId].bombCount = state->agents[state->ourId].maxBombCount - ammo; //should be already OK from bomb enumeration
        state->agents[state->ourId].bombStrength = blast_strength - 1;

        int i, j;
        for (i = 0; i < state->bombs.count-1; i++)
            for (j = 0; j < state->bombs.count-1-i; j++)
                if (BMB_TIME(state->bombs[j]) > BMB_TIME(state->bombs[j+1]))
                    std::swap(state->bombs[j], state->bombs[j+1]);
    }

    void Environment::MakeGameFromPython_dortmund(bool agent0Alive, bool agent1Alive, bool agent2Alive, bool agent3Alive, uint8_t * board, double * bomb_life,
                                                 double * bomb_blast_strength, int posx, int posy, int blast_strength, bool can_kick, int ammo, int teammate_id)
    {
        util::TickFlames(*state);
        state->agents[0].bombCount = 0;
        state->agents[1].bombCount = 0;
        state->agents[2].bombCount = 0;
        state->agents[3].bombCount = 0;
        state->agents[0].diedAt = state->timeStep;
        state->agents[1].diedAt = state->timeStep;
        state->agents[2].diedAt = state->timeStep;
        state->agents[3].diedAt = state->timeStep;
        state->agents[0].dead = !agent0Alive;
        state->agents[1].dead = !agent1Alive;
        state->agents[2].dead = !agent2Alive;
        state->agents[3].dead = !agent3Alive;

        auto previousBombs = state->bombs;
        state->bombs.count = 0;

        for(int i=0; i < state->powerup_incr.count; i++)
        {
            if(board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] >= PyAGENT0)
                state->agents[board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] - PyAGENT0].bombStrength++;
            else if(board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] == PyFOG)
                board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] = PyINCRRANGE; //so it will be added again
        }
        for(int i=0; i < state->powerup_extrabomb.count; i++)
        {
            if(board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] >= PyAGENT0)
                state->agents[board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] - PyAGENT0].maxBombCount++;
            else if(board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] == PyFOG)
                board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] = PyEXTRABOMB; //so it will be added again
        }
        for(int i=0; i < state->powerup_kick.count; i++)
        {
            if(board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] >= PyAGENT0)
                state->agents[board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] - PyAGENT0].canKick = true;
            else if(board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] == PyFOG)
                board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] = PyKICK; //so it will be added again
        }
        for(int i=0; i < state->woods.count; i++)
        {
            if(board[state->woods[i].x + state->woods[i].y * 11] == PyFOG)
                board[state->woods[i].x + state->woods[i].y * 11] = PyWOOD; //so it will be added again
        }

        state->woods.count = 0;
        state->powerup_incr.count = 0;
        state->powerup_kick.count = 0;
        state->powerup_extrabomb.count = 0;
        //state->flames.count = 0; //keep!
        state->agents[0].x = -1;
        state->agents[0].y = -1;
        state->agents[1].x = -1;
        state->agents[1].y = -1;
        state->agents[2].x = -1;
        state->agents[2].y = -1;
        state->agents[3].x = -1;
        state->agents[3].y = -1;

        state->agents[0].extraBombPowerupPoints = 0;
        state->agents[0].extraRangePowerupPoints = 0;
        state->agents[0].firstKickPowerupPoints = 0;
        state->agents[0].otherKickPowerupPoints = 0;
        state->agents[1].extraBombPowerupPoints = 0;
        state->agents[1].extraRangePowerupPoints = 0;
        state->agents[1].firstKickPowerupPoints = 0;
        state->agents[1].otherKickPowerupPoints = 0;
        state->agents[2].extraBombPowerupPoints = 0;
        state->agents[2].extraRangePowerupPoints = 0;
        state->agents[2].firstKickPowerupPoints = 0;
        state->agents[2].otherKickPowerupPoints = 0;
        state->agents[3].extraBombPowerupPoints = 0;
        state->agents[3].extraRangePowerupPoints = 0;
        state->agents[3].firstKickPowerupPoints = 0;
        state->agents[3].otherKickPowerupPoints = 0;
        state->agents[0].woodDemolished = 0;
        state->agents[1].woodDemolished = 0;
        state->agents[2].woodDemolished = 0;
        state->agents[3].woodDemolished = 0;
        state->timeStep++;
        if (teammate_id == 9)
        {
            //this means that we are in ffa mode!
            throw std::runtime_error("ffa mode not supported");
        }
        state->teammateId = teammate_id - 10;

        if(state->ourId == 1 || state->ourId == 3)
        {
            state->enemy1Id = 0;
            state->enemy2Id = 2;
        }else
        {
            state->enemy1Id = 1;
            state->enemy2Id = 3;
        }

        //std::cout << state->ourId << " " << state->teammateId << " : " << state->enemy1Id << " " << state->enemy2Id << std::endl;

        for(int i=0; i<11*11; i++)
        {
            int y = i/11;
            int x = i%11;
            switch(board[i])
            {
                case PyPASSAGE:
                    state->board[y][x] = PASSAGE;
                    break;
                case PyRIGID:
                    state->board[y][x] = RIGID;
                    break;
                case PyWOOD:
                    state->board[y][x] = WOOD;
                    {Position p; p.x = x; p.y = y; state->woods.NextPos() = p; state->woods.count++;}
                    break;
                case PyBOMB:
                    //Already known bombs are handled, hopefully for them:
                    //  BMB_TIME(state->bombs[?]) == bomb_life[i]
                    //  BMB_STRENGTH(state->bombs[?]) == bomb_blast_strength[i] - 1
                    //New bombs are born under an agent, and added there. so here in theory we only handle bombs which were placed in fog and discovered later
                    state->board[y][x] = BOMB;
                    {

                        Bomb *b = &state->bombs.NextPos();
                        SetBombID(*b, 5); //Unknown bomb
                        SetBombPosition(*b, x, y);
                        SetBombStrength(*b, (int)bomb_blast_strength[i] - 1);
                        //SetBombVelocity(*b, 0);
                        SetBombTime(*b, bomb_life[i]);
                        //state->agents[bombId].bombCount++;
                        state->bombs.count++;
                    }
                    break;
                case PyFLAMES:
                    if(!IS_FLAME(state->board[y][x]))
                        state->board[y][x] = FLAMES;
                    //else: don't delete the set FLAME-ID
                    break;
                case PyFOG:
                    if(state->board[y][x] >= AGENT0)
                        state->board[y][x] = FOG;
                    /*if(state->board[y][x] == RIGID)
                        state->board[y][x] = RIGID;
                    else
                        state->board[y][x] = FOG;*/
                    break;
                case PyEXTRABOMB:
                    state->board[y][x] = EXTRABOMB;
                    {Position p; p.x = x; p.y = y; state->powerup_extrabomb.NextPos() = p; state->powerup_extrabomb.count++;}
                    break;
                case PyINCRRANGE:
                    state->board[y][x] = INCRRANGE;
                    {Position p; p.x = x; p.y = y; state->powerup_incr.NextPos() = p; state->powerup_incr.count++;}
                    break;
                case PyKICK:
                    state->board[y][x] = KICK;
                    {Position p; p.x = x; p.y = y; state->powerup_kick.NextPos() = p; state->powerup_kick.count++;}
                    break;
                case PyAGENTDUMMY:
                    state->board[y][x] = AGENTDUMMY;
                    break;
                case PyAGENT0:
                case PyAGENT1:
                case PyAGENT2:
                case PyAGENT3:
                    state->board[y][x] = AGENT0 + (board[i] - PyAGENT0);
                    state->agents[board[i] - PyAGENT0].x = x;
                    state->agents[board[i] - PyAGENT0].y = y;
                    if((int)bomb_blast_strength[i] > 0) //hidden bomb under us
                    {
                        int bombId = board[i] - PyAGENT0;//the agent who placed it
                        Bomb *b = &state->bombs.NextPos();
                        SetBombID(*b, bombId);
                        SetBombPosition(*b, x, y);
                        SetBombStrength(*b, (int)bomb_blast_strength[i] - 1);
                        SetBombVelocity(*b, 0);
                        SetBombTime(*b, bomb_life[i]);
                        if (bomb_life[i] > 8) //Let's assume it was placed by the guy on the top
                        {
                            //extra info: the agent's bomb-strength
                            state->agents[bombId].bombStrength = (int)bomb_blast_strength[i] - 1;
                        }
                        state->bombs.count++;
                    }
                    break;
                default:
                    std::cout << "Unknown item in py board: " << board[i] << std::endl;
            }
        }

        for(int bomb_index=0; bomb_index < previousBombs.count; bomb_index++) {
            if (BMB_TIME(previousBombs[bomb_index]) == 1) {
                //exploded since that

                state->SpawnFlame_passive(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), BMB_STRENGTH(previousBombs[bomb_index]));

                previousBombs.RemoveAt(bomb_index);
                bomb_index--;
                continue;
            }
            Position originalPosition; originalPosition.x = BMB_POS_X(previousBombs[bomb_index]); originalPosition.y = BMB_POS_Y(previousBombs[bomb_index]);
            Position expectedPosition = bboard::util::DesiredPosition(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), (bboard::Move) BMB_VEL(previousBombs[bomb_index]));
            if(_CheckPos_basic(state.get(), expectedPosition.x, expectedPosition.y))
            {
                //Could move
                if(board[expectedPosition.y * BOARD_SIZE + expectedPosition.x] == PyFOG) {
                    if(bomb_blast_strength[originalPosition.y * BOARD_SIZE + originalPosition.x] - 1 == BMB_STRENGTH(previousBombs[bomb_index]) &&
                        bomb_life[originalPosition.y * BOARD_SIZE + originalPosition.x] == BMB_TIME(previousBombs[bomb_index]) - 1)
                    {
                        for(int bomb_index2=0; bomb_index2 < state->bombs.count; bomb_index2++) {
                            if(BMB_POS_X(state->bombs[bomb_index2]) == originalPosition.x && BMB_POS_Y(state->bombs[bomb_index2]) == originalPosition.y)
                            {
                                SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
                                SetBombVelocity(state->bombs[bomb_index2], 0);
                                break;
                            }
                        }

                    }else {
                        state->bombs.AddElem(previousBombs[bomb_index]);
                        SetBombTime(state->bombs[state->bombs.count - 1], BMB_TIME(previousBombs[bomb_index]) - 1);
                    }
                }else if(board[expectedPosition.y * BOARD_SIZE + expectedPosition.x] == PyFLAMES)
                {
                    state->SpawnFlame_passive(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), BMB_STRENGTH(previousBombs[bomb_index]));

                }else if(bomb_blast_strength[expectedPosition.y * BOARD_SIZE + expectedPosition.x] > 0) //Not good to filter for AGENT or BOMB, because AGENT can be here if it kicked, without BOMB
                {
                    bool found = false;
                    for(int bomb_index2=0; bomb_index2 < state->bombs.count; bomb_index2++) {
                        if(BMB_POS_X(state->bombs[bomb_index2]) == expectedPosition.x && BMB_POS_Y(state->bombs[bomb_index2]) == expectedPosition.y)
                        {
                            found = true;
                            SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
                            SetBombVelocity(state->bombs[bomb_index2], BMB_VEL(previousBombs[bomb_index]));
                            break;
                        }
                    }
                }else{
                    //Ups, it is not at the expected position, maybe somebody kicked it??

                    int moves[]{1,2,3,4};
                    bool found = false;
                    for(int move : moves)
                    {
                        Position kickedToPosition = bboard::util::DesiredPosition(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), (bboard::Move) move);
                        if(_CheckPos_basic(state.get(), kickedToPosition.x, kickedToPosition.y) && bomb_blast_strength[kickedToPosition.y * BOARD_SIZE + kickedToPosition.x] > 0) {
                            if (bomb_life[kickedToPosition.y * BOARD_SIZE + kickedToPosition.x] == BMB_TIME(previousBombs[bomb_index]) - 1)
                            {
                                for(int bomb_index2=0; bomb_index2 < state->bombs.count; bomb_index2++) {
                                    if(BMB_POS_X(state->bombs[bomb_index2]) == kickedToPosition.x && BMB_POS_Y(state->bombs[bomb_index2]) == kickedToPosition.y)
                                    {
                                        found = true;
                                        SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
                                        SetBombVelocity(state->bombs[bomb_index2], move);
                                        break;
                                    }
                                }
                                break;
                            }
                        }
                    }
                }

            }else{
                //Couldn't move
                if(state->board[BMB_POS_Y(previousBombs[bomb_index])][BMB_POS_X(previousBombs[bomb_index])] == FOG)
                {
                    //SetBombVelocity(previousBombs[bomb_index], 0); //Stopping
                    state->bombs.AddElem(previousBombs[bomb_index]);
                    SetBombTime(state->bombs[state->bombs.count - 1], BMB_TIME(previousBombs[bomb_index])-1);
                }else if(state->board[BMB_POS_Y(previousBombs[bomb_index])][BMB_POS_X(previousBombs[bomb_index])] == BOMB)
                {
                    bool found = false;
                    for(int bomb_index2=0; bomb_index2 < state->bombs.count; bomb_index2++) {
                        if(BMB_POS(state->bombs[bomb_index2]) == BMB_POS(previousBombs[bomb_index]))
                        {
                            found = true;
                            //SetBombVelocity(state->bombs[bomb_index2], 0); //Stopping
                            SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
                            break;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < state->bombs.count; i++) {
            if (BMB_ID_KNOWN(state->bombs[i]))
                state->agents[BMB_ID(state->bombs[i])].bombCount++;
        }

        for (int i = 0; i < state->bombs.count-1; i++) {
            for (int j = 0; j < state->bombs.count - 1 - i; j++)
                if (BMB_TIME(state->bombs[j]) > BMB_TIME(state->bombs[j + 1]))
                    std::swap(state->bombs[j], state->bombs[j + 1]);
        }


        state->agents[state->enemy1Id].maxBombCount = std::max(state->agents[state->enemy1Id].maxBombCount, state->agents[state->enemy1Id].bombCount);
        state->agents[state->enemy2Id].maxBombCount = std::max(state->agents[state->enemy2Id].maxBombCount, state->agents[state->enemy2Id].bombCount);
        state->agents[state->teammateId].maxBombCount = std::max(state->agents[state->teammateId].maxBombCount, state->agents[state->teammateId].bombCount);

        state->aliveAgents = int(agent0Alive) + int(agent1Alive) + int(agent2Alive) + int(agent3Alive);
        state->agents[state->ourId].x = posy;
        state->agents[state->ourId].y = posx;
        state->agents[state->ourId].canKick = can_kick;
        state->agents[state->ourId].bombCount = state->agents[state->ourId].maxBombCount - ammo; //should be already OK from bomb enumeration
        state->agents[state->ourId].bombStrength = blast_strength - 1;

    }

	void Environment::MakeGameFromPython_eisenach(bool agent0Alive, bool agent1Alive, bool agent2Alive, bool agent3Alive, uint8_t * board, double * bomb_life,
		double * bomb_blast_strength, int posx, int posy, int blast_strength, bool can_kick, int ammo, int teammate_id)
	{
		util::TickFlames(*state);
		state->agents[0].bombCount = 0;
		state->agents[1].bombCount = 0;
		state->agents[2].bombCount = 0;
		state->agents[3].bombCount = 0;
		state->agents[0].diedAt = state->timeStep;
		state->agents[1].diedAt = state->timeStep;
		state->agents[2].diedAt = state->timeStep;
		state->agents[3].diedAt = state->timeStep;
		state->agents[0].dead = !agent0Alive;
		state->agents[1].dead = !agent1Alive;
		state->agents[2].dead = !agent2Alive;
		state->agents[3].dead = !agent3Alive;

		auto previousBombs = state->bombs;
		state->bombs.count = 0;

		for (int i = 0; i < state->powerup_incr.count; i++)
		{
			if (board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] >= PyAGENT0)
				state->agents[board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] - PyAGENT0].bombStrength++;
			else if (board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] == PyFOG)
				board[state->powerup_incr[i].x + state->powerup_incr[i].y * 11] = PyINCRRANGE; //so it will be added again
		}
		for (int i = 0; i < state->powerup_extrabomb.count; i++)
		{
			if (board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] >= PyAGENT0)
				state->agents[board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] - PyAGENT0].maxBombCount++;
			else if (board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] == PyFOG)
				board[state->powerup_extrabomb[i].x + state->powerup_extrabomb[i].y * 11] = PyEXTRABOMB; //so it will be added again
		}
		for (int i = 0; i < state->powerup_kick.count; i++)
		{
			if (board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] >= PyAGENT0)
				state->agents[board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] - PyAGENT0].canKick = true;
			else if (board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] == PyFOG)
				board[state->powerup_kick[i].x + state->powerup_kick[i].y * 11] = PyKICK; //so it will be added again
		}
		for (int i = 0; i < state->woods.count; i++)
		{
			if (board[state->woods[i].x + state->woods[i].y * 11] == PyFOG)
				board[state->woods[i].x + state->woods[i].y * 11] = PyWOOD; //so it will be added again
		}

		state->woods.count = 0;
		state->powerup_incr.count = 0;
		state->powerup_kick.count = 0;
		state->powerup_extrabomb.count = 0;
		//state->flames.count = 0; //keep!
		state->agents[0].x = -1;
		state->agents[0].y = -1;
		state->agents[1].x = -1;
		state->agents[1].y = -1;
		state->agents[2].x = -1;
		state->agents[2].y = -1;
		state->agents[3].x = -1;
		state->agents[3].y = -1;

        state->agents[0].extraBombPowerupPoints = 0;
        state->agents[0].extraRangePowerupPoints = 0;
        state->agents[0].firstKickPowerupPoints = 0;
        state->agents[0].otherKickPowerupPoints = 0;
        state->agents[1].extraBombPowerupPoints = 0;
        state->agents[1].extraRangePowerupPoints = 0;
        state->agents[1].firstKickPowerupPoints = 0;
        state->agents[1].otherKickPowerupPoints = 0;
        state->agents[2].extraBombPowerupPoints = 0;
        state->agents[2].extraRangePowerupPoints = 0;
        state->agents[2].firstKickPowerupPoints = 0;
        state->agents[2].otherKickPowerupPoints = 0;
        state->agents[3].extraBombPowerupPoints = 0;
        state->agents[3].extraRangePowerupPoints = 0;
        state->agents[3].firstKickPowerupPoints = 0;
        state->agents[3].otherKickPowerupPoints = 0;
		state->agents[0].woodDemolished = 0;
		state->agents[1].woodDemolished = 0;
		state->agents[2].woodDemolished = 0;
		state->agents[3].woodDemolished = 0;
		state->timeStep++;
		if (teammate_id == 9)
		{
			//this means that we are in ffa mode!
			throw std::runtime_error("ffa mode not supported");
		}
		state->teammateId = teammate_id - 10;

		if (state->ourId == 1 || state->ourId == 3)
		{
			state->enemy1Id = 0;
			state->enemy2Id = 2;
		}
		else
		{
			state->enemy1Id = 1;
			state->enemy2Id = 3;
		}

		//std::cout << state->ourId << " " << state->teammateId << " : " << state->enemy1Id << " " << state->enemy2Id << std::endl;

		for (int i = 0; i < 11 * 11; i++)
		{
			int y = i / 11;
			int x = i % 11;
			switch (board[i])
			{
			case PyPASSAGE:
				state->board[y][x] = PASSAGE;
				break;
			case PyRIGID:
				state->board[y][x] = RIGID;
				break;
			case PyWOOD:
				state->board[y][x] = WOOD;
				{Position p; p.x = x; p.y = y; state->woods.NextPos() = p; state->woods.count++; }
				break;
			case PyBOMB:
				//Already known bombs are handled, hopefully for them:
				//  BMB_TIME(state->bombs[?]) == bomb_life[i]
				//  BMB_STRENGTH(state->bombs[?]) == bomb_blast_strength[i] - 1
				//New bombs are born under an agent, and added there. so here in theory we only handle bombs which were placed in fog and discovered later
				state->board[y][x] = BOMB;
				{

					Bomb *b = &state->bombs.NextPos();
					SetBombID(*b, 5); //Unknown bomb
					SetBombPosition(*b, x, y);
					SetBombStrength(*b, (int)bomb_blast_strength[i] - 1);
					//SetBombVelocity(*b, 0);
					SetBombTime(*b, bomb_life[i]);
					//state->agents[bombId].bombCount++;
					state->bombs.count++;
				}
				break;
			case PyFLAMES:
				if (!IS_FLAME(state->board[y][x]))
					state->board[y][x] = FLAMES;
				//else: don't delete the set FLAME-ID
				break;
			case PyFOG:
				if (state->board[y][x] >= AGENT0)
					state->board[y][x] = FOG;
				/*if(state->board[y][x] == RIGID)
					state->board[y][x] = RIGID;
				else
					state->board[y][x] = FOG;*/
				break;
			case PyEXTRABOMB:
				state->board[y][x] = EXTRABOMB;
				{Position p; p.x = x; p.y = y; state->powerup_extrabomb.NextPos() = p; state->powerup_extrabomb.count++; }
				break;
			case PyINCRRANGE:
				state->board[y][x] = INCRRANGE;
				{Position p; p.x = x; p.y = y; state->powerup_incr.NextPos() = p; state->powerup_incr.count++; }
				break;
			case PyKICK:
				state->board[y][x] = KICK;
				{Position p; p.x = x; p.y = y; state->powerup_kick.NextPos() = p; state->powerup_kick.count++; }
				break;
			case PyAGENTDUMMY:
				state->board[y][x] = AGENTDUMMY;
				break;
			case PyAGENT0:
			case PyAGENT1:
			case PyAGENT2:
			case PyAGENT3:
				state->board[y][x] = AGENT0 + (board[i] - PyAGENT0);
				state->agents[board[i] - PyAGENT0].x = x;
				state->agents[board[i] - PyAGENT0].y = y;
				if ((int)bomb_blast_strength[i] > 0) //hidden bomb under us
				{
					int bombId = board[i] - PyAGENT0;//the agent who placed it
					Bomb *b = &state->bombs.NextPos();
					SetBombID(*b, bombId);
					SetBombPosition(*b, x, y);
					SetBombStrength(*b, (int)bomb_blast_strength[i] - 1);
					SetBombVelocity(*b, 0);
					SetBombTime(*b, bomb_life[i]);
					if (bomb_life[i] > 8) //Let's assume it was placed by the guy on the top
					{
						//extra info: the agent's bomb-strength
						state->agents[bombId].bombStrength = (int)bomb_blast_strength[i] - 1;
					}
					state->bombs.count++;

					std::cout << "Bomb under agent " << board[i] - PyAGENT0 << " at " << y << " " << x << " time: " << bomb_life[i] << std::endl;
				}
				break;
			default:
				std::cout << "Unknown item in py board: " << board[i] << std::endl;
			}
		}

		for (int bomb_index = 0; bomb_index < previousBombs.count; bomb_index++) {
			if (BMB_TIME(previousBombs[bomb_index]) == 1) {
				//exploded since that

				state->SpawnFlame_passive(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), BMB_STRENGTH(previousBombs[bomb_index]));

				previousBombs.RemoveAt(bomb_index);
				bomb_index--;
				continue;
			}
			Position originalPosition; originalPosition.x = BMB_POS_X(previousBombs[bomb_index]); originalPosition.y = BMB_POS_Y(previousBombs[bomb_index]);
			Position expectedPosition = bboard::util::DesiredPosition(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), (bboard::Move) BMB_VEL(previousBombs[bomb_index]));
			if (_CheckPos_basic(state.get(), expectedPosition.x, expectedPosition.y))
			{
				//Could move
				if (board[expectedPosition.y * BOARD_SIZE + expectedPosition.x] == PyFOG) {
					if (bomb_blast_strength[originalPosition.y * BOARD_SIZE + originalPosition.x] - 1 == BMB_STRENGTH(previousBombs[bomb_index]) &&
						bomb_life[originalPosition.y * BOARD_SIZE + originalPosition.x] == BMB_TIME(previousBombs[bomb_index]) - 1)
					{
						std::cout << "Bomb couldn't enter fog at " << originalPosition.y << " " << originalPosition.y;

						for (int bomb_index2 = 0; bomb_index2 < state->bombs.count; bomb_index2++) {
							if (BMB_POS_X(state->bombs[bomb_index2]) == originalPosition.x && BMB_POS_Y(state->bombs[bomb_index2]) == originalPosition.y)
							{
								if (BMB_TIME(state->bombs[bomb_index2]) != BMB_TIME(previousBombs[bomb_index]) - 1)
									std::cout << "Unexpected STOP-FOG bomb time at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << std::endl;
								else
									std::cout << "Matched old and new STOP-FOG bomb at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << std::endl;

								SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
								SetBombVelocity(state->bombs[bomb_index2], 0);
								break;
							}
						}

					}
					else {
						state->bombs.AddElem(previousBombs[bomb_index]);
						SetBombTime(state->bombs[state->bombs.count - 1], BMB_TIME(previousBombs[bomb_index]) - 1);
						std::cout << "Bomb known from memory in fog at " << BMB_POS_Y(state->bombs[state->bombs.count - 1]) << " " << BMB_POS_X(state->bombs[state->bombs.count - 1]) << std::endl;
					}
				}
				else if (board[expectedPosition.y * BOARD_SIZE + expectedPosition.x] == PyFLAMES)
				{
					std::cout << "Bomb exploded probably at " << expectedPosition.y << " " << expectedPosition.x << std::endl;

					state->SpawnFlame_passive(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), BMB_STRENGTH(previousBombs[bomb_index]));

				}
				else if (bomb_blast_strength[expectedPosition.y * BOARD_SIZE + expectedPosition.x] > 0) //Not good to filter for AGENT or BOMB, because AGENT can be here if it kicked, without BOMB
				{
					bool found = false;
					for (int bomb_index2 = 0; bomb_index2 < state->bombs.count; bomb_index2++) {
						if (BMB_POS_X(state->bombs[bomb_index2]) == expectedPosition.x && BMB_POS_Y(state->bombs[bomb_index2]) == expectedPosition.y)
						{
							found = true;
							if (BMB_TIME(state->bombs[bomb_index2]) != BMB_TIME(previousBombs[bomb_index]) - 1)
								std::cout << "Unexpected bomb time at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << std::endl;
#ifdef VERBOSE_STATE
							else
								std::cout << "Matched old and new bomb at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << std::endl;
#endif

							SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
							SetBombVelocity(state->bombs[bomb_index2], BMB_VEL(previousBombs[bomb_index]));
							break;
						}
					}

					if (!found)
					{
						std::cout << "Bomb is not found in list around " << expectedPosition.y << " " << expectedPosition.x << std::endl;
					}
				}
				else {
					//Ups, it is not at the expected position, maybe somebody kicked it??
					if (state->board[BMB_POS_Y(previousBombs[bomb_index])][BMB_POS_X(previousBombs[bomb_index])] >= AGENT0)
						std::cout << "An agent is instead of a bomb, maybe he kicked it? at " << BMB_POS_Y(previousBombs[bomb_index]) << " " << BMB_POS_X(previousBombs[bomb_index]) << std::endl;

					int moves[]{ 1,2,3,4 };
					bool found = false;
					for (int move : moves)
					{
						Position kickedToPosition = bboard::util::DesiredPosition(BMB_POS_X(previousBombs[bomb_index]), BMB_POS_Y(previousBombs[bomb_index]), (bboard::Move) move);
						if (_CheckPos_basic(state.get(), kickedToPosition.x, kickedToPosition.y) && bomb_blast_strength[kickedToPosition.y * BOARD_SIZE + kickedToPosition.x] > 0) {
							if (bomb_life[kickedToPosition.y * BOARD_SIZE + kickedToPosition.x] == BMB_TIME(previousBombs[bomb_index]) - 1)
							{
								for (int bomb_index2 = 0; bomb_index2 < state->bombs.count; bomb_index2++) {
									if (BMB_POS_X(state->bombs[bomb_index2]) == kickedToPosition.x && BMB_POS_Y(state->bombs[bomb_index2]) == kickedToPosition.y)
									{
										found = true;
										if (BMB_TIME(state->bombs[bomb_index2]) != BMB_TIME(previousBombs[bomb_index]) - 1)
											std::cout << "Unexpected KICKED bomb time at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << std::endl;
										else
											std::cout << "Matched old and new KICKED bomb at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << " move direction: " << move << std::endl;

										SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
										SetBombVelocity(state->bombs[bomb_index2], move);
										break;
									}
								}
								break;
							}
							else {
								std::cout << "Bomb was maybe kicked, suspected bombs's time is not good, now at " << kickedToPosition.y << " " << kickedToPosition.x << std::endl;
							}
						}
					}

					if (!found)
						std::cout << "Bomb is lost around " << expectedPosition.y << " " << expectedPosition.x << std::endl;
				}

			}
			else {
				//Couldn't move
				if (state->board[BMB_POS_Y(previousBombs[bomb_index])][BMB_POS_X(previousBombs[bomb_index])] == FOG)
				{
					//SetBombVelocity(previousBombs[bomb_index], 0); //Stopping
					state->bombs.AddElem(previousBombs[bomb_index]);
					SetBombTime(state->bombs[state->bombs.count - 1], BMB_TIME(previousBombs[bomb_index]) - 1);
				}
				else if (state->board[BMB_POS_Y(previousBombs[bomb_index])][BMB_POS_X(previousBombs[bomb_index])] == BOMB)
				{
					bool found = false;
					for (int bomb_index2 = 0; bomb_index2 < state->bombs.count; bomb_index2++) {
						if (BMB_POS(state->bombs[bomb_index2]) == BMB_POS(previousBombs[bomb_index]))
						{
							found = true;
							if (BMB_TIME(state->bombs[bomb_index2]) != BMB_TIME(previousBombs[bomb_index]) - 1)
								std::cout << "Unexpected STOPPED bomb time at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << std::endl;
							else
								std::cout << "Matched STOPPED old and new bomb at " << BMB_POS_Y(state->bombs[bomb_index2]) << " " << BMB_POS_X(state->bombs[bomb_index2]) << std::endl;

							//SetBombVelocity(state->bombs[bomb_index2], 0); //Stopping
							SetBombID(state->bombs[bomb_index2], BMB_ID(previousBombs[bomb_index]));
							break;
						}
					}
					if (!found)
					{
						std::cout << "STOPPED Bomb is lost around " << expectedPosition.y << " " << expectedPosition.x << std::endl;
					}
				}
				else {
					std::cout << "STOPPED Bomb is lost around " << expectedPosition.y << " " << expectedPosition.x << std::endl;
				}
			}
		}

		for (int i = 0; i < state->bombs.count; i++) {
			if (BMB_ID_KNOWN(state->bombs[i]))
				state->agents[BMB_ID(state->bombs[i])].bombCount++;
		}

		for (int i = 0; i < state->bombs.count - 1; i++) {
			for (int j = 0; j < state->bombs.count - 1 - i; j++)
				if (BMB_TIME(state->bombs[j]) > BMB_TIME(state->bombs[j + 1]))
					std::swap(state->bombs[j], state->bombs[j + 1]);
		}


		state->agents[state->enemy1Id].maxBombCount = std::max(state->agents[state->enemy1Id].maxBombCount, state->agents[state->enemy1Id].bombCount);
		state->agents[state->enemy2Id].maxBombCount = std::max(state->agents[state->enemy2Id].maxBombCount, state->agents[state->enemy2Id].bombCount);
		state->agents[state->teammateId].maxBombCount = std::max(state->agents[state->teammateId].maxBombCount, state->agents[state->teammateId].bombCount);

		state->aliveAgents = int(agent0Alive) + int(agent1Alive) + int(agent2Alive) + int(agent3Alive);
		state->agents[state->ourId].x = posy;
		state->agents[state->ourId].y = posx;
		state->agents[state->ourId].canKick = can_kick;
		state->agents[state->ourId].bombCount = state->agents[state->ourId].maxBombCount - ammo; //should be already OK from bomb enumeration
		state->agents[state->ourId].bombStrength = blast_strength - 1;

	}


void Environment::StartGame(int timeSteps, bool render, bool stepByStep)
{
    state->timeStep = 0;
    while(!this->IsDone() && state->timeStep < timeSteps)
    {

        if(render)
        {
            Print();

            if(listener)
                listener(*this);

            if(stepByStep)
                Pause(false);
        }
        this->Step(true);
    }
    Print();
    PrintGameResult(*this);
}

void ProxyAct(Move& writeBack, Agent& agent, State& state)
{
    writeBack = agent.act(&state);
}

void CollectMovesAsync(Move m[AGENT_COUNT], Environment& e)
{
    std::thread threads[AGENT_COUNT];
    for(unsigned int i = 0; i < AGENT_COUNT; i++)
    {
        if(!e.GetState().agents[i].dead)
        {
            threads[i] = std::thread(ProxyAct,
                                     std::ref(m[i]),
                                     std::ref(*e.GetAgent(i)),
                                     std::ref(e.GetState()));
        }
    }
    Pause(true); //competitive pause
    for(unsigned int i = 0; i < AGENT_COUNT; i++)
    {
        if(!e.GetState().agents[i].dead)
        {
            threads[i].join();
        }
    }
}

void Environment::Step(bool competitiveTimeLimit)
{
    if(!hasStarted || finished)
    {
        return;
    }

    Move m[AGENT_COUNT];


    if(competitiveTimeLimit)
    {
        CollectMovesAsync(m, *this);
    }
    else
    {
        for(unsigned int i = 0; i < AGENT_COUNT; i++)
        {
            if(!state->agents[i].dead)
            {
                m[i] = agents[i]->act(state.get());
            }
        }
    }


    bboard::Step(state.get(), m);
    state->timeStep++;

    if(state->aliveAgents == 1)
    {
        finished = true;
        for(int i = 0; i < AGENT_COUNT; i++)
        {
            if(!state->agents[i].dead)
            {
                agentWon = i;
                // teamwon = team of agent
            }
        }
    }
    if(state->aliveAgents == 0)
    {
        finished = true;
        isDraw = true;
    }
}

void Environment::Print(bool clear)
{
    if(clear)
        std::cout << "\033c"; // clear console on linux
    PrintState(state.get());
}

State& Environment::GetState() const
{
    return *state.get();
}

Agent* Environment::GetAgent(unsigned int agentID) const
{
    return agents[agentID];
}

void Environment::SetAgents(std::array<Agent*, AGENT_COUNT> agents)
{
    for(unsigned int i = 0; i < AGENT_COUNT; i++)
    {
        agents[i]->id = int(i);
    }
    this->agents = agents;
}

bool Environment::IsDone()
{
    return finished;
}

bool Environment::IsDraw()
{
    return isDraw;
}

int Environment::GetWinner()
{
    return agentWon;
}

void Environment::SetStepListener(const std::function<void(const Environment&)>& f)
{
    listener = f;
}

}
