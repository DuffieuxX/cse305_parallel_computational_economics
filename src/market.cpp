#include "market.hpp"

// #1 Class definitions and constructors

Order::Order(bool buy, double quantity, double price, int agent_id)
    : buy(buy),
      quantity(quantity),
      price(price),
      agent_id(agent_id),
      next(nullptr)
{}

Agent::Agent(Agent_type type,int agent_id,Params params,std::mt19937& rng):
type(type),
agent_id(agent_id),
cash(params.initial_cash),
asset_inventory(params.initial_inventory),
chartist_order_size(uniform(rng, 0.2 * params.order_size, 2.0 * params.order_size)),
aggressiveness(half_normal(rng,0,params.sigma_aggressiveness))
{}


Order_book::Order_book() {
    this->bid_head= new Order(false,0,std::numeric_limits<double>::max(),-1);
    this->ask_head=new Order(false,0,std::numeric_limits<double>::min(),-1);
}

Order_book::~Order_book() {
    Order* current = bid_head;
    while(current != nullptr){
        Order* next = current->next;
        delete current;
        current = next;
    }

    current = ask_head;
    while(current != nullptr){
        Order* next = current->next;
        delete current;
        current = next;
    }
}



Order* Agent::new_order(Market market, Params params){

    if(this->type==Agent::Agent_type::Optimist){
        double limit_price =market.p*(1+this->aggressiveness);
        double quantity =std::min(this->chartist_order_size, this->cash/limit_price);
        bool buy=true;

        Order* new_order = new Order(buy, quantity, limit_price,this->agent_id);
        return new_order;

    }

    else if(this->type==Agent::Agent_type::Pessimist){
        double limit_price =market.p*(1-this->aggressiveness);
        double quantity =std::min(this->chartist_order_size, this->asset_inventory);
        bool buy=false;
        Order* new_order = new Order(buy, quantity, limit_price,this->agent_id);
        return new_order;

    }

    else{
        double m_t= (market.pf-market.p)/market.p;

        if(m_t>=params.m_min){
            bool buy=true;
            double limit_price= market.pf*(1+this->aggressiveness/4);
            double fundamentalist_order_size = this->chartist_order_size * (1.0 + params.gamma_f * std::abs(m_t));
            double quantity = std::min(fundamentalist_order_size, this->cash/limit_price);

            Order* new_order= new Order(buy,quantity,limit_price,this->agent_id);
            return new_order;
        }


        if(m_t<=-params.m_min){
            bool buy=false;
            double limit_price= market.pf*(1-this->aggressiveness/4);
            double fundamentalist_order_size = this->chartist_order_size * (1.0 + params.gamma_f * std::abs(m_t));
            double quantity = std::min(fundamentalist_order_size, this->asset_inventory);

            Order* new_order= new Order(buy,quantity,limit_price,this->agent_id);
            return new_order;
        }

        else{
            Order* new_order= new Order(true,0,0,this->agent_id);
            return new_order;
        }
    }

}

void Order_book::add_order(std::vector<Agent*>& agents, Agent* agent, const Market& market, const Params& params){
    Order* new_order = agent->new_order(market,params);
    bool buy =new_order->buy;
    
    if(buy){
        
        Order* previous_ask = this->ask_head;
        Order* current_ask=previous_ask->next;
        
        while(new_order->quantity>0 && current_ask!=nullptr){

            if(current_ask->price<=new_order->price){
                double price_trade = current_ask->price;
                double quantity_trade = std::min(current_ask->quantity, new_order->quantity);

                new_order->quantity -= quantity_trade;
                current_ask-> quantity -= quantity_trade; 

                agents[current_ask->agent_id]->asset_inventory-=quantity_trade;
                agents[current_ask->agent_id]->cash+=quantity_trade*price_trade;

                agent->asset_inventory+=quantity_trade;
                agent->cash-=quantity_trade*price_trade;

                this->volume+=quantity_trade;
                this->volume_weighted_sum+=quantity_trade * price_trade;

                if(current_ask->quantity==0){
                    previous_ask->next=current_ask->next;
                    current_ask=current_ask->next;
                    }    
                }
            else {
                break;
            }
        }
        if(new_order->quantity>0){

        Order* previous_bid = this->bid_head;
        Order* current_bid=previous_bid->next;

        if (current_bid==nullptr){
            previous_bid->next=new_order;
            return;
        }

        while(current_bid->price>new_order->price && current_bid->next!=nullptr){
        previous_bid=current_bid;
        current_bid=current_bid->next;
        }

        if(current_bid->price<=new_order->price){
            previous_bid->next=new_order;
            new_order->next=current_bid;
        }
        else{
            current_bid->next=new_order;
        }
        }
    }

if(!buy){
    Order* previous_bid = this->bid_head;
    Order* current_bid = previous_bid->next;

    while(new_order->quantity > 0 && current_bid != nullptr){
        if(current_bid->price >= new_order->price){
            double price_trade = current_bid->price;
            double quantity_trade = std::min(current_bid->quantity, new_order->quantity);

            new_order->quantity -= quantity_trade;
            current_bid->quantity -= quantity_trade;

            agents[current_bid->agent_id]->asset_inventory += quantity_trade;
            agents[current_bid->agent_id]->cash -= quantity_trade * price_trade;

            agent->asset_inventory -= quantity_trade;
            agent->cash += quantity_trade * price_trade;

            this->volume+=quantity_trade;
            this->volume_weighted_sum+=quantity_trade * price_trade;

            if(current_bid->quantity == 0){
                previous_bid->next = current_bid->next;
                current_bid = current_bid->next;
            }
        } else {
            break;
        }
    }

    if(new_order->quantity > 0){
        Order* previous_ask = this->ask_head;
        Order* current_ask = previous_ask->next;

        if (current_ask==nullptr){
            previous_ask->next=new_order;
            return;
        }
        while(current_ask->price < new_order->price && current_ask->next != nullptr){
            previous_ask = current_ask;
            current_ask = current_ask->next;
        }
        if(current_ask->price >= new_order->price){
            previous_ask->next = new_order;
            new_order->next = current_ask;
        } else {
            current_ask->next = new_order;
        }
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

