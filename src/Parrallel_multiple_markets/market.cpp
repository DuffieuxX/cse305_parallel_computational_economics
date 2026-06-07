#include "market.hpp"

// #1 Class definitions and constructors

Order::Order(bool buy, double quantity, double price, int agent_id)
    : buy(buy),
      quantity(quantity),
      price(price),
      agent_id(agent_id)
{}

Agent::Agent(int agent_id,Params& params,std::mt19937& rng):
agent_id(agent_id),
cash(params.initial_cash),
asset_inventory(params.vector_initial_inventory),
chartist_order_size(uniform(rng, 0.2 * params.order_size, 2.0 * params.order_size)),
aggressiveness(half_normal(rng,0,params.sigma_aggressiveness))
{
    this->vector_type = std::vector<Agent::Agent_type>(params.nb_markets);
}

Counts::Counts(int asset_id):asset_id(asset_id)
{}

Market:: Market(double p, double p_prev, double pf, int asset_id):p(p), p_prev(p_prev), pf(pf), asset_id(asset_id)
{}

Probs::Probs(int asset_id):asset_id(asset_id)
{}

Order_book::Order_book(Params& params,int asset_id) {
    this->order_storage.reserve(params.N);
    this->bids.reserve(params.N);
    this->asks.reserve(params.N);

    this->asset_id=asset_id;
}
void Order_book::reset() {
    order_storage.clear();
    bids.clear();
    asks.clear();
    bid_head = 0;
    ask_head = 0;
    volume = 0;
    volume_weighted_sum = 0;
}


Order Agent::new_order(Market& market, Params& params){
    int asset_id = market.asset_id;

    if(this->vector_type[asset_id]==Agent::Agent_type::Optimist){
        double limit_price =market.p*(1+this->aggressiveness);
        this->cash_lock.lock();
        double quantity =std::min(this->chartist_order_size, this->cash/limit_price);
        this->cash_lock.unlock();
        bool buy=true;
        return Order(buy, quantity, limit_price, this->agent_id);
    }
    else if(this->vector_type[asset_id]==Agent::Agent_type::Pessimist){
        double limit_price =market.p*(1-this->aggressiveness);
        double quantity =std::min(this->chartist_order_size, this->asset_inventory[asset_id]);
        bool buy=false;
        return Order(buy, quantity, limit_price, this->agent_id);
    }
    else{
        double m_t= (market.pf-market.p)/market.p;

        if(m_t>=params.m_min){
            bool buy=true;
            double limit_price= market.pf*(1+this->aggressiveness/4);
            double fundamentalist_order_size = this->chartist_order_size * (1.0 + params.gamma_f * std::abs(m_t));
            this->cash_lock.lock();
            double quantity = std::min(fundamentalist_order_size, this->cash/limit_price);
            this->cash_lock.unlock();

            return Order(buy,quantity,limit_price,this->agent_id);
        }


        if(m_t<=-params.m_min){
            bool buy=false;
            double limit_price= market.pf*(1-this->aggressiveness/4);
            double fundamentalist_order_size = this->chartist_order_size * (1.0 + params.gamma_f * std::abs(m_t));
            double quantity = std::min(fundamentalist_order_size, this->asset_inventory[asset_id]);

            return Order(buy,quantity,limit_price,this->agent_id);
        }
        else{
            return Order(true,0,0,this->agent_id);
        }
    }
}


void lock_two(Agent* a, Agent* b) {
    if (a->agent_id < b->agent_id) {
        a->cash_lock.lock();
        b->cash_lock.lock();
    } else {
        b->cash_lock.lock();
        a->cash_lock.lock();
    }
}

void unlock_two(Agent* a, Agent* b) {
   if (a->agent_id < b->agent_id) {
        a->cash_lock.unlock();
        b->cash_lock.unlock();
    } else {
        b->cash_lock.unlock();
        a->cash_lock.unlock();
    }
}

void Order_book::add_order(std::vector<Agent*>& agents, const Order& new_order_) {
    Order new_order = new_order_;

    int new_order_agent_id = new_order.agent_id;
    int asset_id =this-> asset_id;

    Agent* agent_new_order=agents[new_order_agent_id];

    if(new_order.quantity < 1e-9){
        return;
    }

    bool buy = new_order.buy;
    if(buy){

        while(new_order.quantity > 0.0 && this->ask_head < this->asks.size()){
            Order* best_ask = this->asks[this->ask_head];
            Agent* agent_best_ask=agents[best_ask->agent_id];

            if(best_ask->price<=new_order.price){
                
                lock_two(agent_new_order, agent_best_ask);

                if(agent_new_order-> cash<1e-9){
                    unlock_two(agent_new_order, agent_best_ask);
                    return;
                }

                double price_trade = best_ask->price;
                double quantity_trade = std::min({best_ask->quantity, new_order.quantity, agent_new_order-> cash/price_trade});

                new_order.quantity -= quantity_trade;
                best_ask-> quantity -= quantity_trade; 

                agent_best_ask->asset_inventory[asset_id]-=quantity_trade;
                agent_best_ask->cash+=quantity_trade*price_trade;
            
                agent_new_order->asset_inventory[asset_id]+=quantity_trade;
                agent_new_order->cash-=quantity_trade*price_trade;
                
                this->volume+=quantity_trade;
                this->volume_weighted_sum+=quantity_trade * price_trade;

                unlock_two(agent_new_order, agent_best_ask);

                if (best_ask->quantity < 1e-9){
                    ++this->ask_head;
                }

            }
            else{
                break;
            }
            
        }
        if(new_order.quantity>0){
            this->order_storage.push_back(new_order);
            Order* stored_order = &this->order_storage.back();
            auto first_active = this->bids.begin() + static_cast<std::ptrdiff_t>(this->bid_head);
            auto pos = std::upper_bound(first_active, this->bids.end(), stored_order,
                [](const Order* value, const Order* element) {return value->price > element->price;}
            );
            this->bids.insert(pos, stored_order);
        } 
    }
    else if(!buy){

        while (new_order.quantity > 1e-9 && this->bid_head < this->bids.size()) {
            Order* best_bid = this->bids[this->bid_head];
            Agent* agent_best_bid = agents[best_bid->agent_id];

            if (best_bid->price >= new_order.price) {

                lock_two(agent_new_order, agent_best_bid);

                if (agent_best_bid->cash < 1e-9) {
                    unlock_two(agent_new_order, agent_best_bid);
                    ++this->bid_head;
                    continue;
                }

                double price_trade = best_bid->price;
                double quantity_trade = std::min({best_bid->quantity, new_order.quantity, agent_best_bid->cash /price_trade});

                new_order.quantity -= quantity_trade;
                best_bid->quantity -= quantity_trade;

                agents[best_bid->agent_id]->asset_inventory[asset_id] += quantity_trade;
                
                agents[best_bid->agent_id]->cash -= quantity_trade * price_trade;

                agent_new_order->asset_inventory[asset_id] -= quantity_trade;
                agent_new_order->cash += quantity_trade * price_trade;
                


                this->volume += quantity_trade;
                this->volume_weighted_sum += quantity_trade * price_trade;

                unlock_two(agent_new_order, agent_best_bid);

                if (best_bid->quantity < 1e-9) {
                    ++this->bid_head;
                }
            } else {
                break;
            }
        }

        if (new_order.quantity > 0.0) {
            this->order_storage.push_back(new_order);
            Order* stored_order = &this->order_storage.back();
            auto first_active = this->asks.begin() + static_cast<std::ptrdiff_t>(this->ask_head);
            auto pos = std::upper_bound(first_active, this->asks.end(), stored_order,
                [](const Order* value, const Order* element) { return value->price < element->price;}
            );
            this->asks.insert(pos, stored_order);
        }
    }
}


// #2 Utility functions

double switch_probability(double nu, double U, double dt) {
    return nu * std::exp(U) * dt;
}

double uniform01(std::mt19937& rng) {
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    return dist(rng);
}

double uniform(std::mt19937& rng, double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(rng);
}

double normal(std::mt19937& rng, double mean, double sigma_squared) {
    std::normal_distribution<double> dist(mean, sigma_squared);
    return dist(rng);
}

double half_normal(std::mt19937& rng, double mean, double sigma_squared) {
    std::normal_distribution<double> dist(mean, sigma_squared);
    return std::abs(dist(rng));
}

