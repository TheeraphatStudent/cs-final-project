from datetime import date

class NameManagement:

  def __init__(self, *args,):
    
    now = date.today().isoformat()
    
    self.filename = f"my_model_{now}"
    self.scalername = f"my_scalername_{now}"

  def getFileName(self):
    return self.filename
  
  def getScalerName(self):
    return self.scalername
  