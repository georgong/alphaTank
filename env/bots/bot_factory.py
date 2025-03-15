from env.bots.strategy_bot import SmartStrategyBot
from env.bots.simple_bots import RandomBot, AggressiveBot, DefensiveBot

class BotFactory:
    """Factory class to create different types of bots"""
    
    BOT_TYPES = {
        'smart': SmartStrategyBot,
        'random': RandomBot,
        'aggressive': AggressiveBot,
        'defensive': DefensiveBot
    }
    
    @staticmethod
    def create_bot(bot_type: str, tank):
        """Create a bot of the specified type
        
        Args:
            bot_type (str): Type of bot to create ('smart', 'random', 'aggressive', 'defensive')
            tank: Tank object the bot will control
            
        Returns:
            BaseBot: An instance of the specified bot type
            
        Raises:
            ValueError: If bot_type is not recognized
        """
        if bot_type not in BotFactory.BOT_TYPES:
            raise ValueError(f"Unknown bot type: {bot_type}. Valid types are: {list(BotFactory.BOT_TYPES.keys())}")
            
        return BotFactory.BOT_TYPES[bot_type](tank) 