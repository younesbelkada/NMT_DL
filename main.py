
from agents.main_agent import MainAgent
from hparams import Parameters

params = Parameters.parse()
agent = MainAgent(params)

agent.run()

#train_model(train_iterator, [valid_iterator], shallow_transformer_model_preLN,epochs=1,checkpoint_path='models')
