#include "market.hpp"

// #1 Class definitions and constructors
Agent::Agent(Agent_type type,int agent_id,Params& params,std::mt19937& rng):
type(type),
agent_id(agent_id),
cash(params.initial_cash),
asset_inventory(params.initial_inventory),
chartist_order_size(uniform(rng, 0.2 * params.order_size, 2.0 * params.order_size)),
aggressiveness(half_normal(rng,0,params.sigma_aggressiveness))
{}

Order_book::Order_book( Params& params) {
    this->bids.reserve(params.N);
    this->asks.reserve(params.N);
}

Order_book::~Order_book(){
    for(Order* o : bids) delete o;
    for(Order* o : asks) delete o;
}

void Order_book::add_order(Agent* agents,Order* new_order){
    bool buy = new_order->buy;

    if (new_order->quantity == 0) {
        delete new_order;
        return;
    }
    
    if(buy){

        while(new_order->quantity>0 && !asks.empty()){
            Order* best_ask = this->asks[0];

            if(best_ask->price<=new_order->price){
                double price_trade = best_ask->price;
                double quantity_trade = std::min(best_ask->quantity, new_order->quantity);

                new_order->quantity -= quantity_trade;
                best_ask-> quantity -= quantity_trade; 

                agents[best_ask->agent_id].asset_inventory -= quantity_trade;
                agents[best_ask->agent_id].cash += quantity_trade * price_trade;


                agents[new_order->agent_id].asset_inventory+=quantity_trade;
                agents[new_order->agent_id].cash-=quantity_trade*price_trade;

                this->volume+=quantity_trade;
                this->volume_weighted_sum+=quantity_trade * price_trade;

                if(best_ask->quantity==0){
                    delete best_ask;
                    asks.erase(asks.begin());
                }    
            } else{
                break;
            }
    
        }
        if(new_order->quantity>0){
            auto pos = std::lower_bound(this->bids.begin(), this->bids.end(), new_order,
            [](const Order* a, const Order* b){ return a->price > b->price; });
            bids.insert(pos, new_order);
        }
        else {
            delete new_order;
        }

    }

    if(!buy){

        while(new_order->quantity>0 && !bids.empty()){

            Order* best_bid = this->bids[0];


            if(best_bid->price>=new_order->price){
                double price_trade = best_bid->price;
                double quantity_trade = std::min(best_bid->quantity, new_order->quantity);

                new_order->quantity -= quantity_trade;
                best_bid-> quantity -= quantity_trade; 

                agents[best_bid->agent_id].asset_inventory+=quantity_trade;
                agents[best_bid->agent_id].cash-=quantity_trade*price_trade;

                agents[new_order->agent_id].asset_inventory-=quantity_trade;
                agents[new_order->agent_id].cash+=quantity_trade*price_trade;

                this->volume+=quantity_trade;
                this->volume_weighted_sum+=quantity_trade * price_trade;

                if(best_bid->quantity==0){
                    delete best_bid;
                    bids.erase(bids.begin());
                }    
            }
            else{
                break;
            }
        }
        if(new_order->quantity>0){
                auto pos = std::lower_bound(this->asks.begin(), this->asks.end(), new_order,
                [](const Order* a, const Order* b){ return a->price < b->price; });
                asks.insert(pos, new_order);
        } else {
            delete new_order;
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

