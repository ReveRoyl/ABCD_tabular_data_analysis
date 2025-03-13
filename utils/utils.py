import os
import re
import subprocess
import time
import cx_Oracle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator

# --------------------------------------------------------------------------------------------------

def get_cbcl_details(cbcl_item):
    """
    Get detailed information from the element.html file based on the provided cbcl_q field combination (e.g., "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p").

    Parameters:
        cbcl_item (str): The cbcl_q field combination to look up (e.g., "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p").

    Returns:
        str: Combined detailed information, or "N/A" if not found.
    """
    # Parse the element.html file
    with open(
        r"G:\ABCD\script\trail\trail_tsne_RF\factor analysis\data\element.html",
        "r",
        encoding="utf-8",
    ) as file:
        soup = BeautifulSoup(file, "html.parser")

    # Use regular expressions to extract all cbcl_q fields
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")
    cbcl_items = cbcl_pattern.findall(cbcl_item)

    # Store detailed information for each cbcl field
    details = []

    for cbcl in cbcl_items:
        # Find <td> tags in the HTML that contain the cbcl field
        target = soup.find(
            lambda tag: tag.name == "td" and cbcl in tag.get_text(strip=True)
        )

        # Get detailed information
        if target:
            detail_info = target.find_next("td").get_text(strip=True)
            details.append(detail_info)
        else:
            details.append("N/A")

    # Combine all detailed information into a single string
    combined_details = "; ".join(details) if details else "N/A"

    return combined_details


if __name__ == "__main__":
    # Example call
    detail = get_cbcl_details("avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p")
    print("Detailed information:", detail)


# --------------------------------------------------------------------------------------------------


def translate_text(df, language):
    """ 
    Input DataFrame df (in the form of Row_Name1, Row_Name2, as the number of factors)and language (string)
    
    Returns: a DataFrame containing translated detailed information 
    """

    assert os.path.exists(
        '../data/element.html'
    ), "element directory not found. Make sure you're running this code from the root directory of the project."

    # Parse the element.html file to get column names and detailed information
    with open(
        r"../data/element.html",
        "r",
        encoding="utf-8",
    ) as file:
        soup = BeautifulSoup(file, "html.parser")

    # Create a dictionary to store column names and corresponding detailed information
    result_df = pd.DataFrame()

    # Regular expression to extract cbcl_q column names
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")

    for i in range(0, len(df.columns)):
        # Filter out loading values that meet the criteria
        # factor_values = df[f"Factor {i}"][df[f"Factor {i}"] > 0.1]

        original_text = []
        translated_text = []
        for column_name in df.iloc[:, i]:
            # Find all cbcl_q fields in the column_name
            cbcl_items = cbcl_pattern.findall(
                column_name
            )  # Extract all substrings that match the cbcl_qXX_p or cbcl_qXXh_p format

            # Initialize a list to store detailed information for each cbcl field
            original = []
            details = []
            for cbcl_item in cbcl_items:
                # Get detailed information for each cbcl field
                target = soup.find(
                    lambda tag: tag.name == "td"
                    and cbcl_item in tag.get_text(strip=True)
                )
                if target:
                    detail_info = target.find_next("td").get_text(strip=True)
                    # Save original detailed information
                    original.append(detail_info)

                    # Translate detailed information and add to the result
                    try:
                        translated_detail = GoogleTranslator(
                            source="es", target=language
                        ).translate(detail_info)
                    except AttributeError as e:
                        print(f"An error occurred: {e}")
                        translated_detail = detail_info
                    details.append(translated_detail)
                    time.sleep(0.25)

            # Combine all details into a single string and add to the list
            original_text.append("; ".join(original) if original else "N/A")
            translated_text.append("; ".join(details) if details else "N/A")
        # Create a temporary DataFrame to save factor names, column names, loading values, and detailed information
        temp_df = pd.DataFrame(
            {
                # f"Factor {i} Variable": factor_values.index,  # Store column names
                # f"Factor {i} Loading": factor_values.values,  # Store loading values
                f"Factor {i} Detail": original_text,  # Map detailed information
                f"Factor {i} Translated_Detail": translated_text,  # Map translated detailed information
            }
        )

        # Sort by loading values in descending order
        # sorted_df = temp_df.sort_values(by=f"Factor {i} Loading", ascending=False).reset_index(drop=True)
        # Merge the temporary DataFrame into the result DataFrame
        result_df = pd.concat(
            [result_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1
        )
    return result_df


# Get the raw fMRI data with nda-tool; after creating package in NDA
class GetfMRIdata:
    def __init__(self, package_id, password):
        user = f"k21116947_{package_id}"
        dsn = cx_Oracle.makedsn(
            "mindarvpc.cqahbwk3l1mb.us-east-1.rds.amazonaws.com",
            1521,
            service_name="ORCL",
        )
        self.conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)
        self.s3_samples = []

    def fetch_data(self):
        cursor = self.conn.cursor()
        query = """
        SELECT ENDPOINT
        FROM S3_LINKS
        WHERE ENDPOINT LIKE '%baseline%' AND ENDPOINT LIKE '%rsfMRI%' AND ENDPOINT LIKE '%NDARINV005V6D2C%' AND ENDPOINT LIKE '%MPROC%' 
        """
        cursor.execute(query)
        self.s3_samples = [row[0] for row in cursor.fetchall()]
        cursor.close()

    def save_data(self):
        if not self.s3_samples:
            self.fetch_data()
        np.savetxt("data/s3_links.txt", self.s3_samples, fmt="%s")
        # Assuming `downloadcmd` is a command-line tool you want to run
        try:
            subprocess.run(
                [
                    "downloadcmd",
                    "-dp",
                    "1236370",
                    "-t",
                    "data/s3_links.txt",
                    "-d",
                    "./data/fMRI_data",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running downloadcmd: {e}")

    def close_connection(self):
        self.conn.close()

    def run_all(self):
        self.fetch_data()
        self.save_data()
        self.close_connection()
