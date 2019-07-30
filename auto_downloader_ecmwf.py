from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotVisibleException
from requests import get
import time
from urllib.request import urlretrieve
from threading import Thread

"""
ECMWF의 서버 문제로 죽거나 가끔 페이지 로딩이 느려 delay 타임보다 초과되는 경우에 대해서 고려하지 않음
"""


def wait_and_get_element_css_selector(driver, css_selector, delay=60):
    try:
        return WebDriverWait(driver, delay).until(
            expected_conditions.presence_of_element_located(
                (By.CSS_SELECTOR, css_selector)
            )
        )
    except ElementNotVisibleException:
        print("Can't find element")
        pass


def wait_and_get_element_id(driver, id, delay=60):
    try:
        return WebDriverWait(driver, delay).until(
            expected_conditions.presence_of_element_located(
                (By.ID, id)
            )
        )
    except ElementNotVisibleException:
        print("Can't find element")
        pass


def wait_and_get_element_xpath(driver, name, value, delay=60):
    try:
        return WebDriverWait(driver, delay).until(
            expected_conditions.presence_of_element_located(
                (By.XPATH, "//input[@name='{}' and @value='{}']".format(name, value))
            )
        )
    except ElementNotVisibleException:
        print("Can't find element")
        pass


def wait_until_mouse_clickable(driver, delay=60):
    try:
        WebDriverWait(driver, delay).until(
            driver.find_
        )
    except ElementNotVisibleException:
        print("Can't find element")
        pass


def http_response_check(url):
    request = get(url)
    while request.status_code is not 200:
        print('http status: ' + str(request.status_code))
        request = get(url)


def download_start(url, file_name):
    urlretrieve(url, '/home/seok/nc_file/{}'.format(file_name))


def download_finish(download_thread):
    download_thread.join()
    print('{} download finsh'.format(download_thread.name))


start_year = 2000
end_year = 2018
ecmwf_id = None
ecmwf_pw = '0!tjrdudtn!0'

# Open WebBrowser
driver = webdriver.Chrome()
driver.implicitly_wait(1)
driver.get('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')
wait_and_get_element_css_selector(driver, 'body > div.ui-dialog.ui-widget.ui-widget-content.ui-corner-all.ui-front.ui-dialog-buttons.ui-draggable > div.ui-dialog-titlebar.ui-widget-header.ui-corner-all.ui-helper-clearfix.ui-draggable-handle > button').click()
wait_and_get_element_css_selector(driver, '#header-user > li:nth-child(2) > a').click()

# Login
if ecmwf_id is None or ecmwf_pw is None:
    raise ValueError('Initialize ecmwf_id, ecmwf_pw with your personal account')

wait_and_get_element_id(driver, 'uid').send_keys(ecmwf_id)
wait_and_get_element_id(driver, 'password').send_keys(ecmwf_pw)
wait_and_get_element_id(driver, 'submit-login-password').click()

for year in range(start_year, end_year+1):
    for month in range(1, 12+1):
        print("Start download {} year {} month data".format(year, month))
        # Reload foam page
        driver.implicitly_wait(1)
        driver.get('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')
        wait_and_get_element_css_selector(driver, 'body > div.ui-dialog.ui-widget.ui-widget-content.ui-corner-all.ui-front.ui-dialog-buttons.ui-draggable > div.ui-dialog-titlebar.ui-widget-header.ui-corner-all.ui-helper-clearfix.ui-draggable-handle > button').click()

        # Click date information
        wait_and_get_element_xpath(driver, 'date_year_month', '{}'.format(str(year)+str(month).zfill(2))).click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # Click all time step
        wait_and_get_element_css_selector(driver, '#time > div.footer > a.select_all').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # Click step information
        wait_and_get_element_xpath(driver, 'step', '0').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # 10 metre U wind component
        wait_and_get_element_xpath(driver, 'param', '165.128').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # 10 metre V wind component
        wait_and_get_element_xpath(driver, 'param', '166.128').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # Mean wave direction
        wait_and_get_element_xpath(driver, 'param', '230.140').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # Mean wave period
        wait_and_get_element_xpath(driver, 'param', '232.140').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # Sea surface temperature
        wait_and_get_element_xpath(driver, 'param', '34.128').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # Singificant height of combined wind waves and swell
        wait_and_get_element_xpath(driver, 'param', '229.140').click()
        http_response_check('https://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/')

        # wait_and_get_element_css_selector(driver, '#param > div.footer > a.select_all').click()
        wait_and_get_element_css_selector(driver, '#requestForm > button:nth-child(11)').click()
        wait_and_get_element_css_selector(driver, '#selectorForm > button').click()

        # Check queue status until completion
        status = wait_and_get_element_id(driver, 'jobstatus').text
        while status != 'complete':
            time.sleep(1)
            status = wait_and_get_element_id(driver, 'jobstatus').text

        # Download nc file
        file_url = wait_and_get_element_css_selector(driver, '#jobresults > div > a').get_attribute('href')
        download_thread = Thread(target=download_start, args=(file_url, '{}_{}.nc'.format(year, month)), name='{}_{} nc file'.format(year, month))
        download_thread.start()
        download_finish_thread = Thread(target=download_finish, args=(download_thread,))
        download_finish_thread.start()