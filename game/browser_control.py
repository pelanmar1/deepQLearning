import selenium
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

driver = None

def init():
    global driver 
    driver = webdriver.Chrome("/Users/pelanmar1/Coding/Python/Selenium/chromedriver")
    # driver = webdriver.Firefox(executable_path=r"/Users/pelanmar1/Coding/Python/Selenium/geckodriver")
    driver.get("http://2048game.com/")
    wait = WebDriverWait(driver, 10)

    # Load scripts
    r = driver.execute_script(open("/Users/pelanmar1/Coding/Python/ML/QL/deep-q-learning/game/funcs.js").read())
    print(r)
    print('COLOCA LA VENTANA DEL JUEGO EN LA MITAD IZQUIERDA DE LA PANTALLA Y PRESIONA ENTER')
    pause = input('') #This will wait until you press enter before it continues with the program
    print('INICI')
    return driver

def wait(secs=10):
    print("Waiting")
    WebDriverWait(driver, secs)


def get_grid():
    grid = driver.execute_script("return getGrid();")
    return grid

def get_score():
    score = driver.execute_script("return getScore();")
    return score

def get_best_score():
    best_score = driver.execute_script("return getBestScore();")
    return best_score

def is_over():
    over = driver.execute_script("return isOver();")
    return over

#driver.close()