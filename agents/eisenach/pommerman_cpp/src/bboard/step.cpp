#include <iostream>

#include "bboard.hpp"
#include "step_utility.hpp"

namespace bboard
{


bool Step(State* state, Move* moves)
{
    ///////////////////////
    // Flames, Explosion //
    ///////////////////////

    util::TickFlames(*state);

    ///////////////////////
    //  Player Movement  //
    ///////////////////////

    Position destPos[AGENT_COUNT];
    util::FillDestPos(state, moves, destPos);
    util::FixSwitchMove(state, destPos);
    bool any_switch = util::FixSwitchMove(state, destPos);
    // if all the agents successfully moved to their (initial) destination
    bool agentMoveSuccess = !any_switch;
    util::MoveBombs(state, destPos);

    int dependency[AGENT_COUNT] = {-1, -1, -1, -1};
    int roots[AGENT_COUNT] = {-1, -1, -1, -1};

    // the amount of chain roots
    const int rootNumber = util::ResolveDependencies(state, destPos, dependency, roots);
    const bool ouroboros = rootNumber == 0; // ouroboros formation?

    int rootIdx = 0;
    int i = rootNumber == 0 ? 0 : roots[0]; // no roots -> start from 0

    // iterates 4 times but the index i jumps around the dependencies
    for(int _ = 0; _ < AGENT_COUNT; _++, i = dependency[i])
    {
        if(i == -1)
        {
            rootIdx++;
            i = roots[rootIdx];
        }
        if(i < 0 || i > 3)
        {
            std::cout << "ERROR in step.cpp Step(), indexing agent with " << i << std::endl;
            agentMoveSuccess = false;
            break;
        }
        const Move m = moves[i];

        if(state->agents[i].dead || m == Move::IDLE || state->agents[i].x < 0)
        {
            continue;
        }
        else if(m == Move::BOMB)
        {
            state->PlantBomb(state->agents[i].x, state->agents[i].y, i);
            continue;
        }


        int x = state->agents[i].x;
        int y = state->agents[i].y;

        Position desired = destPos[i];

        if(util::IsOutOfBounds(desired))
        {
            continue;
        }

        int itemOnDestination = state->board[desired.y][desired.x];

        //if ouroboros, the bomb will be covered by an agent
        if(ouroboros)
        {
            for(int j = 0; j < state->bombs.count; j++)
            {
                if(BMB_POS_X(state->bombs[j]) == desired.x
                        && BMB_POS_Y(state->bombs[j]) == desired.y)
                {
                    itemOnDestination = Item::BOMB;
                    break;
                }
            }
        }

        if(IS_FLAME(itemOnDestination))
        {
            state->Kill(i);
            if(state->board[y][x] == Item::AGENT0 + i)
            {
                if(state->HasBomb(x, y))
                {
                    state->board[y][x] = Item::BOMB;
                }
                else
                {
                    state->board[y][x] = Item::PASSAGE;
                }

            }
            continue;
        }
        if(util::HasDPCollision(*state, destPos, i))
        {
            continue;
        }

        //
        // All checks passed - you can try a move now
        //


        // Collect those sweet power-ups
        if(IS_POWERUP(itemOnDestination))
        {
            util::ConsumePowerup(*state, i, itemOnDestination);
            itemOnDestination = 0;
        }

        if(itemOnDestination == Item::BOMB && state->agents[i].canKick)
        {
            //This part is not perfect. If 'Agent0(Right) Bomb(Standing) Agent1(Right)' is the setup, and Agent0 is simulated first, Agent0 can't kick the bomb, because Agent1 is there. It's not enough to check if Agent1's destination is elsewhere because maybe can't move there. In reality, all can move right.
            Position bombDestPos = bboard::util::DesiredPosition(desired.x, desired.y, moves[i]);
            if (bboard::_CheckPos_any(state, bombDestPos.x, bombDestPos.y)) {
                for(int bombi=0; bombi<state->bombs.count; bombi++)
                {
                    if(BMB_POS_X(state->bombs[bombi]) == desired.x && BMB_POS_Y(state->bombs[bombi]) == desired.y)
                    {
                        bool explodes = IS_FLAME(state->board[bombDestPos.y][bombDestPos.x]);
                        SetBombPosition(state->bombs[bombi], bombDestPos.x, bombDestPos.y);
                        SetBombVelocity(state->bombs[bombi], (int)moves[i]);
                        state->board[bombDestPos.y][bombDestPos.x] = BOMB;
                        state->board[desired.y][desired.x] = Item::AGENT0 + i;
                        state->agents[i].x = desired.x;
                        state->agents[i].y = desired.y;

                        // only override the position I came from if it has not been
                        // overridden by a different agent that already took this spot
                        if(state->board[y][x] == Item::AGENT0 + i)
                        {
                            if(state->HasBomb(x, y))
                            {
                                state->board[y][x] = Item::BOMB;
                            }
                            else
                            {
                                state->board[y][x] = Item::PASSAGE;
                            }
                        }

                        if(explodes)
                        {
                            state->ExplodeBomb(bombi);
                            bombi--;
                        }

                        break;
                    }
                }
            }//else: Agent can't kick the bomb
        } else

        // execute move if the destination is free
        // (in the rare case of ouroboros, make the move even
        // if an agent occupies the spot)
        if(itemOnDestination == Item::PASSAGE)
            //GM: If there is a bomb at ouroboros, some agents move, some can't, so they end up on the same position and program crashes.
            //Now ouroboros will not move, which is invalid, but happens rarely and doesn't crash.
                //|| (ouroboros && itemOnDestination >= Item::AGENT0))
        {
            // only override the position I came from if it has not been
            // overridden by a different agent that already took this spot
            if(state->board[y][x] == Item::AGENT0 + i)
            {
                if(state->HasBomb(x, y))
                {
                    state->board[y][x] = Item::BOMB;
                }
                else
                {
                    state->board[y][x] = 0;
                }
            }
            state->board[desired.y][desired.x] = Item::AGENT0 + i;
            state->agents[i].x = desired.x;
            state->agents[i].y = desired.y;
        }
    }

    util::TickBombs(*state);

    return agentMoveSuccess;

}

}
